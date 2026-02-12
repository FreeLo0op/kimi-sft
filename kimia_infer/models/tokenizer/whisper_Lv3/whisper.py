import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from subprocess import CalledProcessError, run, Popen, PIPE
import os
import time
from functools import lru_cache
from typing import Optional, Union
from .modeling_whisper import WhisperModel
# from modeling_whisper import WhisperModel
# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 120
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def load_bytesio_audio(content, sr: int = SAMPLE_RATE):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        "pipe:",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "pipe:",
    ]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=-1)
    out, _ = p.communicate(input=content)
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = 128) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz")  # todo
        # os.path.join("assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # Batch-safe normalization: avoid cross-sample coupling when audio is batched.
    if log_spec.dim() == 3:
        sample_peak = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, sample_peak - 8.0)
    else:
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class WhisperEncoder(nn.Module):
    def __init__(
        self, model_path, mel_batch_size=40, unfreeze_online_whisper_model=False
    ):
        super().__init__()
        self.speech_encoder = WhisperModel.from_pretrained(model_path).encoder
        self.unfreeze_online_whisper_model = unfreeze_online_whisper_model
        if not self.unfreeze_online_whisper_model:
            self.speech_encoder.eval()
        self.mel_batch_size = mel_batch_size

    def forward(self, audio, kimia_whisper_clip_silence=False):
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 2: # (B, T)
                # Handle batch input directly
                return self.forward_batch(audio, kimia_whisper_clip_silence)
            
            audio = audio[0]
            if audio.dtype == torch.bfloat16:
                audio = audio.float().cpu().numpy()
            else:
                audio = audio.cpu().numpy()
        time_step = 0
        audios = []
        # slice audio into CHUNK_LENGTH-second segments using constants
        while time_step * SAMPLE_RATE < audio.shape[0]:
            start = time_step * SAMPLE_RATE
            end = (time_step + CHUNK_LENGTH) * SAMPLE_RATE
            audio_segment = audio[start:end]
            audios.append(audio_segment)
            time_step += CHUNK_LENGTH

        final_audio_embedding = []

        # expected number of mel frames for a full chunk
        expected_frames = N_SAMPLES // HOP_LENGTH
        for audio_segment in audios:
            # import pdb; pdb.set_trace()
            assert audio_segment.shape[0] <= N_SAMPLES
            L = audio_segment.shape[0]
            token_len = (L - 1) // (HOP_LENGTH * 8) + 1  # use HOP_LENGTH constant for robustness
            pad_audio = pad_or_trim(audio_segment.flatten())
            mel = log_mel_spectrogram(pad_audio)  # torch.Size([80, expected_frames])
            assert mel.shape[1] == expected_frames
            if kimia_whisper_clip_silence:
                input_seq_lens_list = [token_len * 4]
                input_seq_lens = torch.LongTensor(input_seq_lens_list).to(
                    torch.cuda.current_device()
                )
                audio_embedding = self.speech_encoder(
                    mel.unsqueeze(0).to(torch.cuda.current_device()).to(torch.bfloat16),
                    return_dict=True,
                    input_seq_lens=input_seq_lens,
                ).last_hidden_state
            else:
                audio_embedding = self.speech_encoder(
                    mel.unsqueeze(0).to(torch.cuda.current_device()).to(torch.bfloat16),
                    return_dict=True,
                ).last_hidden_state
                # audio_embedding: [1, 3000, 1280]
                audio_embedding = audio_embedding[:, : token_len * 4, :]

            final_audio_embedding.append(audio_embedding)

        final_audio_embedding = torch.cat(final_audio_embedding, dim=1)
        return final_audio_embedding

    def forward_batch(self, audio_batch, kimia_whisper_clip_silence=False):
        """
        Optimized batch forward with Dynamic Padding.
        audio_batch: (B, T) tensor on GPU
        """
        B, L = audio_batch.shape
        CHUNK_SAMPLES = 480000 
        hop_length = 160
        
        if kimia_whisper_clip_silence:
            raise NotImplementedError("Batch processing with clip_silence not optimized yet.")
        
        if L <= CHUNK_SAMPLES:
            mel = log_mel_spectrogram(audio_batch, n_mels=self.speech_encoder.config.num_mel_bins, padding=0)

            # padding_needed = CHUNK_SAMPLES - L
            # if padding_needed > 0:
            #     audio_input = F.pad(audio_batch, (0, padding_needed))
            # else:
            #     audio_input = audio_batch
            # mel = log_mel_spectrogram(audio_input, n_mels=self.speech_encoder.config.num_mel_bins, padding=0)

            outputs = self.speech_encoder(mel.to(torch.bfloat16), return_dict=True).last_hidden_state
        else:
            # Long audio path (Chunking)
            # Pad to multiple of 30s
            pad_len = (CHUNK_SAMPLES - (L % CHUNK_SAMPLES)) % CHUNK_SAMPLES
            if pad_len > 0:
                audio_batch_padded = F.pad(audio_batch, (0, pad_len))
            else:
                audio_batch_padded = audio_batch
                
            # Reshape to (B * NumChunks, 30s)
            num_chunks = audio_batch_padded.shape[1] // CHUNK_SAMPLES
            audio_reshaped = audio_batch_padded.reshape(B * num_chunks, CHUNK_SAMPLES)
            
            # Process chunks
            mel = log_mel_spectrogram(audio_reshaped, n_mels=self.speech_encoder.config.num_mel_bins, padding=0)
            outputs = self.speech_encoder(mel.to(torch.bfloat16), return_dict=True).last_hidden_state
            
            # Unpack back to (B, TotalTime, Dim)
            _, T_out, D_out = outputs.shape
            outputs = outputs.reshape(B, num_chunks, T_out, D_out)
            outputs = outputs.flatten(1, 2) # (B, N*1500, D)

        # Enforce valid length (multiple of 4) for both paths
        # This ensures compatibility with downstream reshape(..., T//4, D*4)
        token_len = (L - 1) // (hop_length * 8) + 1
        valid_len = token_len * 4
        
        if outputs.shape[1] > valid_len:
            outputs = outputs[:, :valid_len, :]
        elif outputs.shape[1] < valid_len:
            pad_amt = valid_len - outputs.shape[1]
            outputs = F.pad(outputs, (0, 0, 0, pad_amt))
            
        return outputs

    @torch.no_grad()
    def tokenize_waveform(self, audio, kimia_whisper_clip_silence=False):
        audio_embedding = self.forward(audio, kimia_whisper_clip_silence)
        # audio_embedding = audio_embedding.cpu()
        return audio_embedding

if __name__ == '__main__':
    audio = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/wavs/wavs_batch1/17582032991901418352967560978432.wav'
    audio_waveform = load_audio(audio)
    model = WhisperEncoder(model_path='/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/PaMLLM/PaMLLM_kimi_v2.3/model_infer_ad4/whisper-large-v3', unfreeze_online_whisper_model=False)
    device = 'cuda:0'
    model = model.to(device)
    model = model.bfloat16()
    model.eval()

    audio_embedding1 = model.tokenize_waveform(audio_waveform, kimia_whisper_clip_silence=False)

    audio2 = '/mnt/pfs_l2/jieti_team/SFT/hupeng/data/en/audio_detect/train/output_normalized.wav'
    audio_waveform2 = load_audio(audio2)
    audio_embedding2 = model.tokenize_waveform(audio_waveform2, kimia_whisper_clip_silence=False)

    print(audio_embedding1)
    print(audio_embedding2)
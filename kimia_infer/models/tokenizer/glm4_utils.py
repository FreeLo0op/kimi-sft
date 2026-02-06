import os
import io
import glob
import math
import tarfile
import torch
import torchaudio
import safetensors
from .glm4.speech_tokenizer.configuration_whisper import WhisperVQConfig
from .glm4.speech_tokenizer.modeling_whisper import WhisperVQEncoder, WhisperVQForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast


def load_quantize_encoder(model_path):
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


import torch
import torchaudio
import safetensors
import torch.nn.functional as F
from .glm4.speech_tokenizer.configuration_whisper import WhisperVQConfig
from .glm4.speech_tokenizer.modeling_whisper import WhisperVQEncoder, WhisperVQForConditionalGeneration
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast
from kimia_infer.models.tokenizer.whisper_Lv3.whisper import log_mel_spectrogram

def load_quantize_encoder(model_path):
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


def extract_speech_token(model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, utts):
    dtype = model.conv1.weight.dtype
    device = torch.cuda.current_device()
    n_mels = getattr(feature_extractor, "feature_size", getattr(feature_extractor, "n_mels", 128))
    
    # Constants for 30s chunking @ 16kHz
    SAMPLE_RATE = 16000
    CHUNK_LENGTH = 30 * SAMPLE_RATE
    
    with torch.no_grad():
        # 1. Prepare chunks on GPU
        audio_chunks = []
        indices = []
        chunk_masks = []
        
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            else:
                audio, sample_rate = torchaudio.load(utt)
            
            # Ensure GPU Tensor
            if not isinstance(audio, torch.Tensor):
                audio = torch.tensor(audio)
            audio = audio.to(device)

            # Resample needed?
            if sample_rate != SAMPLE_RATE:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=SAMPLE_RATE
                    ).to(device)
                audio = _resample_buffer[sample_rate](audio)
            
            # Handle channels
            if audio.ndim > 1:
                audio = audio[0]

            # Slice into 30s chunks
            length = audio.shape[0]
            time_step = 0
            while time_step < length:
                end = min(time_step + CHUNK_LENGTH, length)
                segment = audio[time_step: end]
                
                # Create mask (1 for valid audio)
                mask = torch.ones(segment.shape[0], dtype=torch.long, device=device)
                
                # Do NOT pad to 30s yet. Keep original length segment.
                # We will pad dynamically in the batch loop.
                
                audio_chunks.append(segment)
                chunk_masks.append(mask)
                indices.append(idx)
                
                time_step += CHUNK_LENGTH
        
        # 2. Batch Inference
        batch_size = 128
        all_speech_tokens = [[] for _ in range(len(utts))]
        
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride_conv = model.conv1.stride[0] * model.conv2.stride[0]
        # stride = stride_conv * pooling_kernel_size * feature_extractor.hop_length
        
        # Let's perform Log Mel Spectrogram on GPU
        num_chunks = len(audio_chunks)
        for start in range(0, num_chunks, batch_size):
            end = min(start + batch_size, num_chunks)
            
            # Dynamic Padding for the current batch
            batch_segments = audio_chunks[start:end]
            batch_masks = chunk_masks[start:end]
            
            max_len_in_batch = max([seg.shape[0] for seg in batch_segments])
            
            # Pad segments to max_len_in_batch
            # Use a pre-allocated tensor for speed
            padded_audio = torch.zeros((len(batch_segments), max_len_in_batch), dtype=batch_segments[0].dtype, device=device)
            padded_mask = torch.zeros((len(batch_segments), max_len_in_batch), dtype=batch_masks[0].dtype, device=device)
            
            for i, (seg, msk) in enumerate(zip(batch_segments, batch_masks)):
                padded_audio[i, :seg.shape[0]] = seg
                padded_mask[i, :msk.shape[0]] = msk
            
            # Extract Features (B, n_mels, Frames)
            # log_mel_spectrogram returns (B, n_mels, frames)
            input_features = log_mel_spectrogram(padded_audio, n_mels=n_mels, device=device)
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            
            # Generate Attention Mask (B, Frames) from sample mask
            # Stride = 160 (HOP_LENGTH)
            feature_mask = padded_mask[:, ::160] 
            
            # Align padding
            if feature_mask.shape[1] > input_features.shape[2]:
                feature_mask = feature_mask[:, :input_features.shape[2]]
            elif feature_mask.shape[1] < input_features.shape[2]:
                feature_mask = F.pad(feature_mask, (0, input_features.shape[2] - feature_mask.shape[1]))

            input_features = input_features.to(dtype)
            
            # Strict masking logic
            final_mask = feature_mask[:, ::stride_conv]
            final_mask = final_mask[:, ::pooling_kernel_size]
            
            # Run Model
            outputs = model(input_features=input_features, attention_mask=feature_mask)
            
            speech_tokens = outputs.quantized_token_ids
            
            # Collect valid tokens efficiently
            # Transfer everything to CPU once to avoid sync in loop
            speech_tokens_cpu = speech_tokens.cpu() # (B, T_tokens)
            final_mask_cpu = final_mask.cpu().bool() # (B, T_tokens)
            
            for i in range(len(speech_tokens)):
                idx = indices[start + i]
                # Filter locally on CPU
                toks = speech_tokens_cpu[i]
                mask_i = final_mask_cpu[i]
                
                valid_len = min(mask_i.shape[0], toks.shape[0])
                valid_tokens = toks[:valid_len][mask_i[:valid_len]].tolist()
                
                all_speech_tokens[idx].extend(valid_tokens)

        return all_speech_tokens

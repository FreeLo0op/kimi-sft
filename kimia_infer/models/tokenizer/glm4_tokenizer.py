import torch
import torchaudio
import torchaudio.functional as F
from typing import List, Sequence

from transformers import WhisperFeatureExtractor
from .glm4.speech_tokenizer.modeling_whisper import WhisperVQEncoder
from .glm4_utils import extract_speech_token
from torch import nn


class Glm4Tokenizer(nn.Module):
    def __init__(self, tokenizer_path):
        super().__init__()
        self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    def tokenize(self, speech=None, audio_path=None, sr=16000):
        if audio_path:
            audio, loaded_sr = torchaudio.load(audio_path)
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if loaded_sr != sr:
                audio = F.resample(audio, orig_freq=loaded_sr, new_freq=sr)
            audio_info = (audio, sr)
        else:
            assert speech is not None
            assert sr
            if isinstance(speech, list):
                speech = torch.tensor(speech).unsqueeze(0)
            if len(speech.shape) == 1:
                speech = speech.unsqueeze(0)
            audio_info = (speech, sr)

        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [audio_info]
        )[0]
        audio_tokens = torch.tensor(audio_tokens).unsqueeze(0)
        return audio_tokens

    def tokenize_batch(self, audio_paths:Sequence[str], sr:int=16000) -> List[torch.Tensor]:
        audio_infos = []
        for audio_path in audio_paths:
            audio, loaded_sr = torchaudio.load(audio_path)
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if loaded_sr != sr:
                audio = F.resample(audio, orig_freq=loaded_sr, new_freq=sr)
            audio_infos.append((audio, sr))

        if not audio_infos:
            return []

        with torch.no_grad():
            audio_tokens_batch = extract_speech_token(
                self.whisper_model, self.feature_extractor, audio_infos
            )

        token_tensors: List[torch.Tensor] = []
        for tokens in audio_tokens_batch:
            token_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            token_tensors.append(token_tensor)
        return token_tensors
            

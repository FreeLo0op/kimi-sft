import os
import numpy as np
from typing import List, Dict
import torch
import torchaudio
from loguru import logger
from transformers import AutoTokenizer
import time
from concurrent.futures import ThreadPoolExecutor
import contextlib

from kimia_infer.models.tokenizer.whisper_Lv3.whisper import WhisperEncoder
from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
from kimia_infer.utils.data import KimiAContent
from kimia_infer.utils.special_tokens import instantiate_extra_tokens

class BatchKimiAContent:
    def __init__(
        self, 
        audio_input_ids: torch.Tensor,
        text_input_ids: torch.Tensor,
        is_continuous_mask: torch.Tensor,
        continuous_feature: List[torch.Tensor],
        attention_mask: torch.Tensor
    ):
        self.audio_input_ids = audio_input_ids
        self.text_input_ids = text_input_ids
        self.is_continuous_mask = is_continuous_mask
        self.continuous_feature = continuous_feature
        self.attention_mask = attention_mask

    def to(self, device):
        self.audio_input_ids = self.audio_input_ids.to(device)
        self.text_input_ids = self.text_input_ids.to(device)
        self.is_continuous_mask = self.is_continuous_mask.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.continuous_feature = [f.to(device) for f in self.continuous_feature]
        return self

class KimiAPromptManager:
    def __init__(self, model_path:str, audio_tokenizer:str, kimia_token_offset:int, kimia_text_audiodelaytokens:int, device:str):
        self.device = device
        self.audio_tokenizer = Glm4Tokenizer(audio_tokenizer)
        self.audio_tokenizer = self.audio_tokenizer.to(self.device).bfloat16()

        logger.info(f"Looking for resources in {model_path}")
        logger.info(f"Loading whisper model")

        self.whisper_model = WhisperEncoder(
            os.path.join(model_path, "whisper-large-v3"), mel_batch_size=20
        )
        self.whisper_model = self.whisper_model.to(self.device)
        self.whisper_model = self.whisper_model.bfloat16()
        self.whisper_model.eval()

        logger.info(f"Loading text tokenizer")
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
        else:
            logger.info(f"Can not find text tokenizer in {model_path}, Loading default text tokenizer from moonshotai/Kimi-Audio-7B-Instruct")
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                "/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B-Instruct", trust_remote_code=True
            )
        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)
        self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens
        self.kimia_token_offset = kimia_token_offset
        
        # Initialize CUDA streams for parallel processing
        if torch.cuda.is_available() and "cuda" in str(self.device):
            self.stream_glm4 = torch.cuda.Stream(device=self.device)
            self.stream_whisper = torch.cuda.Stream(device=self.device)
        else:
            self.stream_glm4 = None
            self.stream_whisper = None

    def _load_audio(self, audio_content):
        # 统一处理音频加载，返回[1, T]的Tensor
        if isinstance(audio_content, str):
            wav, sr = torchaudio.load(audio_content)
        elif isinstance(audio_content, torch.Tensor):
            wav, sr = audio_content, 16000
        else: 
            wav, sr = torch.tensor(audio_content), 16000

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        
        if wav.ndim > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
            
        return wav.to(self.device)

    def _tokenize_text(self, text):
        if text is None:
            return None
        token_ids = self.text_tokenizer.encode(text, bos=False, eos=False)
        return token_ids

    def _tokenize_audio(self, wav_tensor):
        # 统一不再做参数检查，假定已是Tensor
        wav_tokens = self.audio_tokenizer.tokenize(audio_tensor=wav_tensor)
        wav_tokens = wav_tokens + self.kimia_token_offset
        wav_tokens_list = wav_tokens.squeeze(0).cpu().numpy().tolist()
        return wav_tokens_list


    def _tokenize_audio_batch(self, wav_paths: List[str], batch_size:int=16) -> List[List[int]]:
        if not wav_paths:
            return []

        if torch.cuda.is_available():
            if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
            elif isinstance(self.device, str) and self.device.startswith('cuda'):
                torch.cuda.set_device(self.device)

        if batch_size and len(wav_paths) > batch_size:
            logger.warning(
                f"Received {len(wav_paths)} wav paths exceeding batch_size {batch_size}, falling back to sequential processing."
            )

        batch_paths = wav_paths
        if hasattr(self.audio_tokenizer, 'tokenize_batch'):
            wav_tokens_batch = self.audio_tokenizer.tokenize_batch(batch_paths)
        else:
            wav_tokens_batch = [
                self.audio_tokenizer.tokenize(audio_path=path) for path in batch_paths
            ]

        batched_tokens: List[List[int]] = []
        for wav_tokens in wav_tokens_batch:
            if isinstance(wav_tokens, torch.Tensor):
                token_tensor = wav_tokens.squeeze(0).to(torch.long)
            else:
                token_tensor = torch.as_tensor(wav_tokens, dtype=torch.long)
            token_tensor = token_tensor + self.kimia_token_offset
            batched_tokens.append(token_tensor.detach().cpu().tolist())

        return batched_tokens

    def extract_whisper_feat(self, wav_tensor: torch.Tensor):
        return self.extract_whisper_feat_batch([wav_tensor])[0]

    def extract_whisper_feat_batch(self, wavs: List[torch.Tensor]):
        if not wavs: return []
        
        # Determine max length for padding input
        # 简化的逻辑：直接处理 tensor 列表
        max_len = max([w.shape[-1] for w in wavs])
        
        padded_wavs = []
        for w in wavs:
            if w.dim() == 2: w = w.squeeze(0)
            diff = max_len - w.shape[-1]
            if diff > 0:
                w = torch.nn.functional.pad(w, (0, diff), value=0.0)
            padded_wavs.append(w)
            
        wav_batch = torch.stack(padded_wavs).to(self.device).bfloat16()
        
        continous_features = self.whisper_model.tokenize_waveform(wav_batch)
        
        continous_features = continous_features.reshape(
            continous_features.shape[0],
            int(continous_features.shape[1] // 4),
            continous_features.shape[2] * 4,
        )
        
        return [f.unsqueeze(0) for f in continous_features] # Ret [1, L, D] list

    def _process_batch_audios(self, chats_list: List[List[Dict]]):
        # 1. Collect all audio tasks
        audio_tasks = [] #(chat_idx, msg_idx, content)
        for i, chats in enumerate(chats_list):
            # Input must contain text and audio, and audio is the sencond element
            audio_path = chats[-1]["content"]
            audio_tasks.append((i, 1, audio_path))

        # 2. Load Audios using unified loader
        audio_tensor = [] # List[Tensor]
        
        for _, _, content in audio_tasks:
            wav = self._load_audio(content)
            audio_tensor.append(wav)

        # 3. Parallel Tokenize (GLM4) & Extract Whisper Feature
        
        # CUDA Events for accurate GPU timing (Async)
        start_evt_whisper = torch.cuda.Event(enable_timing=True)
        end_evt_whisper = torch.cuda.Event(enable_timing=True)

        def _run_glm4(stream):
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            with ctx:
                # Support batch tokenization with tensors
                token_tensors = self.audio_tokenizer.tokenize_batch_tensor(audio_tensor, sr=16000)
                audio_tokens_list = []
                for toks in token_tensors:
                    toks = toks + self.kimia_token_offset
                    # Note: We must move to list here, which involves synchronization/blocking 
                    # if we immediately access values on CPU. 
                    audio_tokens_list.append(toks.squeeze(0).cpu().numpy().tolist())
                
                return audio_tokens_list

        def _run_whisper(stream):
            ctx = torch.cuda.stream(stream) if stream else contextlib.nullcontext()
            with ctx:
                res = self.extract_whisper_feat_batch(audio_tensor)
                return res

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit Whisper (Longer task ~16ms) first to ensure GPU saturation
            future_whisper = executor.submit(_run_whisper, self.stream_whisper)
            # Submit GLM4 (Shorter task ~9ms)
            future_glm4 = executor.submit(_run_glm4, self.stream_glm4)
            
            # Retrieve results
            # GLM4 requires CPU sync inside the function, so this might block the thread
            audio_tokens_list = future_glm4.result()
            # Whisper returns tensors on GPU (stream_whisper)
            whisper_features_list = future_whisper.result()
        
        # Synchronize streams to ensure all GPU work is done before downstream usage
        if self.stream_glm4: self.stream_glm4.synchronize()
        if self.stream_whisper: self.stream_whisper.synchronize()
        
        # 5. Write back to chats_list
        for (chat_i, msg_i, _), tokens, feat in zip(audio_tasks, audio_tokens_list, whisper_features_list):
            chats_list[chat_i][msg_i]["audio_tokens"] = tokens
            chats_list[chat_i][msg_i]["whisper_feature"] = feat

    def tokenize_message(
        self,
        message,
        tokenize_role=True,
        has_ct_token=False,
        has_msg_end_token=False,
        extract_whisper_feature=False,
        output_type: str = "text",
    ):
        kimia_content_msg = KimiAContent()
        role = message["role"]
        has_loss = role == "assistant"

        if tokenize_role:
            if role == "user":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_assistant_msg_start
                )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            else:
                raise NotImplementedError(f"role: {role}")

        if message["message_type"] == "text":
            # text = message["content"]
            text_tokens = self._tokenize_text(message["content"])

            kimia_content_msg.text_extend(text_tokens, has_loss)
            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank] * len(text_tokens)
            )

            if role == "assistant":
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos, has_loss) # eos for text stream
                kimia_content_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)

        elif message["message_type"] == "audio":
            if "audio_tokens" in message:
                speech_tokens = message["audio_tokens"]
            else:
                # 使用统一加载函数
                audio_tensor = self._load_audio(message["content"])
                speech_tokens = self._tokenize_audio(audio_tensor)

            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=True, audio_token_loss_mask=has_loss)
            kimia_content_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=has_loss) # EOS for audio stream
            kimia_content_msg.text_extend(
                [self.extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2)
            )

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(
                        self.extra_tokens.kimia_speech_ctd_id
                    )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            if extract_whisper_feature:
                if "whisper_feature" in message:
                    whisper_feature = message["whisper_feature"]
                else:
                    # Fallback
                    audio_tensor = self._load_audio(message["content"])

                    whisper_feature = self.extract_whisper_feat(audio_tensor)
                
                kimia_content_msg.continuous_feature.append(whisper_feature)
        elif message["message_type"] == "audio-text":
            audio_path, text = message["content"]
            # 统一加载
            wav_tensor = self._load_audio(audio_path)
            speech_tokens = self._tokenize_audio(wav_tensor)
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.audio_extend([self.extra_tokens.kimia_text_blank] * self.kimia_text_audiodelaytokens)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=False)
            kimia_content_msg.text_extend(text_tokens)
            text_pad_tokens = (self.kimia_text_audiodelaytokens + len(speech_tokens) - len(text_tokens)) * [self.extra_tokens.kimia_text_blank]
            kimia_content_msg.text_extend(text_pad_tokens)

        elif message["message_type"] == None:
            pass
        else:
            raise NotImplementedError(f"message_type: {message['message_type']}")

        if has_msg_end_token:
            kimia_content_msg.audio_append(self.extra_tokens.msg_end, audio_token_loss_mask=False)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert (
            kimia_content_msg.is_valid()
        ), f"kimia_content_msg is not valid: {kimia_content_msg}"

        return kimia_content_msg

    def get_prompt(
        self, messages: List[Dict], output_type: str = "text", add_assistant_start_msg: bool = True
    ) -> KimiAContent:
        """
        messages: List[Dict]
        messages[i] = {
            "role": "user" | "assistant" | "system",
            "content": str
        }
        """
        assert output_type in ["text", "both"]

        msgs: List[KimiAContent] = []
        tokenize_role = True
        has_ct_token = False
        has_msg_end_token = False

        previous_role = None
        for msg_idx, message in enumerate(messages):
            if previous_role is None:
                tokenize_role = True
            else:
                if message["role"] == previous_role:
                    tokenize_role = False
                else:
                    tokenize_role = True

            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                if messages[msg_idx + 1]["role"] != message["role"]:
                    has_ct_token = True
                    has_msg_end_token = True
                else:
                    has_ct_token = False
                    has_msg_end_token = False

            previous_role = message["role"]

            msg = self.tokenize_message(
                message=message,
                tokenize_role=tokenize_role,
                has_ct_token=has_ct_token,
                has_msg_end_token=has_msg_end_token,
                extract_whisper_feature=True,
                output_type=output_type,
            )
            msgs.append(msg)

        if add_assistant_start_msg:
            assistant_start_msg = self.tokenize_message(
                    message={
                        "role": "assistant",
                    "message_type": None,
                },
                tokenize_role=True,
                has_ct_token=False,
                has_msg_end_token=False,
            )

            msgs.append(assistant_start_msg)

        ret_msg = msgs[0]

        for msg in msgs[1:]:
            ret_msg.merge(msg)

        return ret_msg

    def collate_content(self, contents: List[KimiAContent]) -> BatchKimiAContent:
        if not contents:
            return None

        # Convert contents to tensors first
        tensor_samples = [c.to_tensor() for c in contents]
        
        # Unpack tensors (input_ids, text_ids, mask, audio_loss_mask, text_loss_mask)
        input_ids_list = [s[0] for s in tensor_samples]
        text_input_ids_list = [s[1] for s in tensor_samples]
        is_continuous_mask_list = [s[2] for s in tensor_samples]
        
        whisper_features = [c.continuous_feature for c in contents] # List of lists of tensors

        # Find max lengths
        max_input_ids_len = max(t.shape[1] for t in input_ids_list)
        max_text_input_ids_len = max(t.shape[1] for t in text_input_ids_list)

        pad_token = self.extra_tokens.pad if hasattr(self.extra_tokens, "pad") else self.text_tokenizer.pad_token_id

        input_ids_batch = []
        text_input_ids_batch = []
        is_continuous_mask_batch = []
        
        all_whisper_features = []
        for feat_list in whisper_features:
            for feat in feat_list:
                all_whisper_features.append(feat)

        # Pad logic (Vectorized Pre-allocation)
        batch_size = len(input_ids_list)
        device = input_ids_list[0].device # Usually CPU here
        
        batch_input_ids = torch.full((batch_size, max_input_ids_len), pad_token, dtype=input_ids_list[0].dtype, device=device)
        batch_is_continuous_mask = torch.full((batch_size, max_input_ids_len), False, dtype=is_continuous_mask_list[0].dtype, device=device)
        batch_text_input_ids = torch.full((batch_size, max_text_input_ids_len), pad_token, dtype=text_input_ids_list[0].dtype, device=device)
        
        for i in range(batch_size):
            # Left Pad input_ids
            width_audio = input_ids_list[i].shape[1]
            if width_audio > 0:
                batch_input_ids[i, -width_audio:] = input_ids_list[i].squeeze(0)
                batch_is_continuous_mask[i, -width_audio:] = is_continuous_mask_list[i].squeeze(0)
            
            # Left Pad text_input_ids
            width_text = text_input_ids_list[i].shape[1]
            if width_text > 0:
                batch_text_input_ids[i, -width_text:] = text_input_ids_list[i].squeeze(0)
        
        # Attention Mask
        attention_mask = (batch_input_ids != pad_token).long()
        
        return BatchKimiAContent(
            audio_input_ids=batch_input_ids,
            text_input_ids=batch_text_input_ids, 
            is_continuous_mask=batch_is_continuous_mask,
            continuous_feature=all_whisper_features,
            attention_mask=attention_mask
        )

    def get_batch_prompt(
        self, chats_list: List[List[Dict]], output_type: str = "text", add_assistant_start_msg: bool = True
    ) -> BatchKimiAContent:
        # Pre-process audios in batch
        self._process_batch_audios(chats_list)
        
        contents = []
        for chats in chats_list:
            content = self.get_prompt(chats, output_type=output_type, add_assistant_start_msg=add_assistant_start_msg)
            contents.append(content)
        
        return self.collate_content(contents)

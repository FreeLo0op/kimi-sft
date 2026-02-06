import os
import tqdm
import time
import torch
from loguru import logger
from huggingface_hub import cached_assets_path
from transformers import AutoModelForCausalLM

from kimia_infer.models.detokenizer import get_audio_detokenizer
from .prompt_manager import KimiAPromptManager
from kimia_infer.utils.sampler import KimiASampler
from huggingface_hub import snapshot_download

class KimiAudio(object):
    def __init__(self, model_path: str, device: str, load_detokenizer: bool = False):
        logger.info(f"Loading kimi-audio main model")
        self.device = device
        if os.path.exists(model_path):
            cache_path = model_path
        else:
            cache_path = snapshot_download(model_path)
    
        self.alm = AutoModelForCausalLM.from_pretrained(
            cache_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.alm = self.alm.to(self.device)

        model_config = self.alm.config
        self.kimia_text_audiodelaytokens = model_config.kimia_mimo_audiodelaytokens
        self.kimia_token_offset = model_config.kimia_token_offset

        self.prompt_manager = KimiAPromptManager(
            model_path=cache_path, 
            audio_tokenizer='/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/THUDM/glm-4-voice-tokenizer',
            kimia_token_offset=self.kimia_token_offset, 
            kimia_text_audiodelaytokens=self.kimia_text_audiodelaytokens,
            device=self.device
        )

        if load_detokenizer:
            logger.info(f"Loading detokenizer")
            vocoder_cache_path = '/mnt/pfs_l2/jieti_team/SFT/hupeng/resources/llm-base-models/Kimi-Audio-7B'
            self.detokenizer = get_audio_detokenizer(vocoder_cache_path)
        else:
            self.detokenizer = None
        self.extra_tokens = self.prompt_manager.extra_tokens
        self.eod_ids = [self.extra_tokens.msg_end, self.extra_tokens.media_end]

    @torch.inference_mode()
    def _generate_text_only(
            self,
            audio_input_ids: torch.Tensor,
            text_input_ids: torch.Tensor,
            max_new_tokens: int = 2048,
            audio_top_k: int = 5,
            audio_temperature: float = 0.0,
            audio_repetition_penalty: float = 1.0,
            audio_repetition_window_size: int = 64,
            text_top_k: int = 5,
            text_temperature: float = 0.0,
            text_repetition_penalty: float = 1.0,
            text_repetition_window_size: int = 16,
            is_continuous_mask: torch.Tensor = None,
            continous_feature: torch.Tensor = None,
            output_type: str = "text",
        ):
            sampler = KimiASampler(
                audio_top_k=audio_top_k,
                audio_temperature=audio_temperature,
                audio_repetition_penalty=audio_repetition_penalty,
                audio_repetition_window_size=audio_repetition_window_size,
                text_top_k=text_top_k,
                text_temperature=text_temperature,
                text_repetition_penalty=text_repetition_penalty,
                text_repetition_window_size=text_repetition_window_size,
            )
            text_previous_tokens = torch.zeros(
                (max_new_tokens,),
                dtype=torch.int,
                device=self.device,
            )
            text_previous_tokens_prob = torch.zeros(
                (max_new_tokens,),
                dtype=torch.float,
                device=self.device,
            )
            
            blank_audio_token_tensor = torch.tensor(
                [self.extra_tokens.kimia_text_blank],
                dtype=torch.int,
                device=self.device
            )
            
            decoder_input_audio_ids = audio_input_ids.clone()
            decoder_input_text_ids = text_input_ids.clone()
            decoder_position_ids = (
                torch.arange(
                    0, decoder_input_audio_ids.shape[1], 
                    device=self.device
                )
                .unsqueeze(0)
                .long()
            )
            
            decoder_input_whisper_feature = continous_feature
            decoder_is_continuous_mask = is_continuous_mask
            past_key_values = None

            last_position_id = decoder_input_audio_ids.shape[1] - 1
            valid_text_length = 0
            
            # 纯文本生成主循环
            for i in range(max_new_tokens):
                # 模型前向传播
                _, text_logits, past_key_values = self.alm.forward(
                    input_ids=decoder_input_audio_ids,
                    text_input_ids=decoder_input_text_ids,
                    whisper_input_feature=decoder_input_whisper_feature,
                    is_continuous_mask=decoder_is_continuous_mask,
                    position_ids=decoder_position_ids,
                    past_key_values=past_key_values,
                    return_dict=False,
                )
                
                next_token_text, max_prob_text = sampler.sample_text_logits(
                    text_logits, recent_tokens=text_previous_tokens[:i] if i > 0 else None
                )
                
                if next_token_text.item() == self.extra_tokens.kimia_text_eos:
                    text_previous_tokens[i:i+1] = next_token_text
                    text_previous_tokens_prob[i:i+1] = max_prob_text
                    valid_text_length = i + 1
                    break
                
                text_previous_tokens[i:i+1] = next_token_text
                text_previous_tokens_prob[i:i+1] = max_prob_text
                valid_text_length += 1
                
                decoder_input_text_ids = next_token_text.unsqueeze(1)
                decoder_input_audio_ids = blank_audio_token_tensor.unsqueeze(1)
                
                # 更新位置ID
                decoder_position_ids = torch.tensor(
                    [[last_position_id + 1]], 
                    device=self.device
                ).long()
                last_position_id += 1
                
                decoder_input_whisper_feature = None
                decoder_is_continuous_mask = None
            
            return_text_tokens = (
                text_previous_tokens[:valid_text_length]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            return_text_tokens_probs = (
                text_previous_tokens_prob[:valid_text_length]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            
            return None, return_text_tokens, return_text_tokens_probs


    @torch.inference_mode()
    def _generate_loop(
        self,
        audio_input_ids: torch.Tensor,
        text_input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 50,
        audio_top_k: int = 5,
        audio_temperature: float = 0.0,
        audio_repetition_penalty: float = 1.0,
        audio_repetition_window_size: int = 64,
        text_top_k: int = 5,
        text_temperature: float = 0.0,
        text_repetition_penalty: float = 1.0,
        text_repetition_window_size: int = 16,
        is_continuous_mask: torch.Tensor = None,
        continous_feature: torch.Tensor = None,
        output_type: str = "text",
    ):
        batch_size = audio_input_ids.shape[0]
        sampler = KimiASampler(
            audio_top_k=audio_top_k,
            audio_temperature=audio_temperature,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
        )

        text_stream_is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        audio_stream_is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device) # Only used for 'both' mode
        
        # Output buffers [Batch, MaxLen]
        previous_audio_tokens = torch.zeros(
            (batch_size, 4096),
            dtype=torch.int,
            device=self.device,
        )
        text_previous_tokens = torch.zeros(
            (batch_size, 4096),
            dtype=torch.int,
            device=self.device,
        )
        text_previous_tokens_prob = torch.zeros(
            (batch_size, 4096),
            dtype=torch.float,
            device=self.device,
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        
        # Position IDs: [Batch, Seq]
        # Assuming Left Padding for input, we want generation to continue from the last non-pad token?
        # Standard approach with Left Padding: position_ids are correct (0..L-1).
        # We construct position_ids based on seq len.
        decoder_position_ids = (
            torch.arange(
                0, decoder_input_audio_ids.shape[1], device=self.device
            )
            .unsqueeze(0)
            .long()
            .repeat(batch_size, 1)
        )
        
        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None

        last_position_id = decoder_input_audio_ids.shape[1] - 1 # This is scalar, but distinct for batches?
        # Actually with Left Padding, we just append to the right. 
        # Position IDs just increment.
        
        # Valid length trackers
        valid_text_length = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        valid_audio_length = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(max_new_tokens):
            # Attention mask needs to be updated? 
            # forward accepts attention_mask. For the first step we pass the padded mask.
            # Subsequent steps: we pass None (if using cache, model usually updates/handles it?)
            # Usually with past_key_values, we just pass new tokens.
            
            # Using custom attention_mask if provided
            current_attention_mask = attention_mask if i == 0 else None
            
            audio_logits, text_logits, past_key_values = self.alm.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask,
                return_dict=False,
            )
            
            # Squeeze dim 1 if present [B, 1, V] -> [B, V]
            if len(text_logits.shape) == 3:
                text_logits = text_logits[:,-1,:]
            if len(audio_logits.shape) == 3:
                audio_logits = audio_logits[:,-1,:]

            recent_text = text_previous_tokens[:, :i] if i > 0 else None
            next_token_text, max_prob_text = sampler.sample_text_logits(
                text_logits, recent_tokens=recent_text
            )
            
            recent_audio = previous_audio_tokens[:, :i] if i > 0 else None
            next_audio_token = sampler.sample_audio_logits(
                audio_logits, recent_tokens=recent_audio
            )

            # Update finished status
            eos_mask = (next_token_text == self.extra_tokens.kimia_text_eos)
            text_stream_is_finished = text_stream_is_finished | eos_mask
            
            # Override finished tokens with Blank/Pad
            next_token_text = torch.where(text_stream_is_finished, 
                                          torch.tensor(self.extra_tokens.kimia_text_blank, device=self.device), 
                                          next_token_text)
            
            # Only increment valid length if not finished (and not just became finished)
            # Actually valid length includes EOS usually? Let's exclude EOS and pads for detokenizer convenience
            # If just became finished (eos_mask is True), we don't count it as "text content" usually?
            # Existing code: if eos, break. 
            # Here parallel: valid_length += 1 where not finished.
            
            # We increment for those who were NOT finished before this step.
            # Actually easier: just store everything and clip later by checking EOS index.
            valid_text_length += (~text_stream_is_finished).long() 
            # Note: if EOS occurs, text_stream_is_finished becomes True. 
            # We want to record EOS? Detokenize usually stops before EOS.
            
            text_previous_tokens[:, i] = next_token_text
            text_previous_tokens_prob[:, i] = max_prob_text

            # Audio Logic
            # Init audio blanking for delay
            if i < self.kimia_text_audiodelaytokens:
                next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self.extra_tokens.kimia_text_blank)
                else:
                    # Check audio EOS
                    # eod_ids is list, convert to tensor for efficient checking
                    # audio_stream_is_finished check
                     is_audio_eos = torch.isin(next_audio_token, torch.tensor(self.eod_ids, device=self.device))
                     audio_stream_is_finished = audio_stream_is_finished | is_audio_eos
                     
                     next_audio_token = torch.where(audio_stream_is_finished,
                                                    torch.tensor(self.extra_tokens.kimia_text_blank, device=self.device),
                                                    next_audio_token)
                     
                     # Increment valid length
                     valid_audio_length += (~audio_stream_is_finished).long()

            previous_audio_tokens[:, i] = next_audio_token

            # Check global finish
            if output_type == "text":
                if text_stream_is_finished.all():
                    break
            elif output_type == "both":
                if text_stream_is_finished.all() and audio_stream_is_finished.all():
                    break
            
            # Prepare next step
            decoder_input_audio_ids = next_audio_token.unsqueeze(1)
            decoder_input_text_ids = next_token_text.unsqueeze(1)
            
            # Update Position IDs
            last_position_id += 1
            decoder_position_ids = (
                    torch.zeros(batch_size, 1, device=self.device)
                    .fill_(last_position_id + 1)
                    .long()
            )
            
            # Update attention_mask if we were explicitly passing it (we passed it 1st step)
            # For next steps with cache, we append 1s to attention mask effectively.
            if attention_mask is not None:
                # We need to maintain the full attention mask [B, L_current]
                attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.device, dtype=torch.long)], dim=1)
                
            decoder_input_whisper_feature = None
            decoder_is_continuous_mask = None

        # Post-Processing: Convert results to List[List]
        return_text_tokens = []
        return_text_tokens_probs = []
        return_audio_tokens = []
        
        for b in range(batch_size):
            t_len = valid_text_length[b]
            a_len = valid_audio_length[b]
            
            return_text_tokens.append(text_previous_tokens[b, :t_len].detach().cpu().numpy().tolist())
            return_text_tokens_probs.append(text_previous_tokens_prob[b, :t_len].detach().cpu().numpy().tolist())
            
            if output_type == "both":
                # Audio start index is kimia_text_audiodelaytokens
                # valid_audio_length counts effective tokens AFTER delay (because we didn't increment during delay)
                # But previous_audio_tokens buffer has delay tokens filled with Blank.
                # Logic above: if i < delay: blank. else: if valid: valid+=1.
                # So we want to return tokens from delay to delay + val_len.
                start = self.kimia_text_audiodelaytokens
                end = start + a_len
                # Check bounds
                end = min(end, max_new_tokens)
                return_audio_tokens.append(previous_audio_tokens[b, start:end].detach().cpu().numpy().tolist())
            else:
                return_audio_tokens.append([])

        return return_audio_tokens, return_text_tokens, return_text_tokens_probs

    @torch.inference_mode()
    def generate(
        self,
        chats: list[dict] | list[list[dict]],
        output_type="text",
        audio_temperature=0.0,
        audio_top_k=5,
        text_temperature=0.0,
        text_top_k=5,
        audio_repetition_penalty=1.0,
        audio_repetition_window_size=64,
        text_repetition_penalty=1.0,
        text_repetition_window_size=16,
        max_new_tokens=2048,
    ):
        ## Check format logic skipped for now
        
        is_single_input = False
        if len(chats) > 0 and isinstance(chats[0], dict):
            chats_batch = [chats]
            is_single_input = True
        else:
            chats_batch = chats

        # Batch processing
        batch_content = self.prompt_manager.get_batch_prompt(chats_batch, output_type=output_type)
        batch_content.to(self.device)
        
        audio_input_ids = batch_content.audio_input_ids
        text_input_ids = batch_content.text_input_ids
        is_continuous_mask = batch_content.is_continuous_mask
        attention_mask = batch_content.attention_mask
        audio_features = batch_content.continuous_feature # This is a flattened list of tensors

        generated_wav_tokens, generated_text_tokens, generated_text_tokens_probs = self._generate_loop(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
            audio_repetition_window_size=audio_repetition_window_size,
            text_top_k=text_top_k,
            text_temperature=text_temperature,
            text_repetition_penalty=text_repetition_penalty,
            text_repetition_window_size=text_repetition_window_size,
            is_continuous_mask=is_continuous_mask,
            continous_feature=audio_features,
            output_type=output_type,
        )

        generated_wavs = []
        generated_texts = []
        
        # Process output batch
        for b in range(len(generated_text_tokens)):
            # Text
            text_toks = [
                t for t in generated_text_tokens[b] if t < self.kimia_token_offset
            ]
            generated_texts.append(self.detokenize_text([text_toks])[0]) # Pass list of 1 list
            
            # Audio
            if output_type == "both":
                wav_toks = [t for t in generated_wav_tokens[b] if t >= self.kimia_token_offset]
                if wav_toks:
                    wav_tensor = torch.tensor(wav_toks).unsqueeze(0) - self.kimia_token_offset
                    if self.detokenizer is not None:
                        # Assuming detokenize_audio can handle 1 item, we loop.
                        # Can optimize later if detokenizer supports batch.
                        gen_wav = self.detokenize_audio(wav_tensor)
                        generated_wavs.append(gen_wav)
                    else:
                        generated_wavs.append(None)
                else:
                    generated_wavs.append(None)
            else:
                generated_wavs.append(None)

        if is_single_input:
            return generated_wavs[0], generated_texts[0], generated_text_tokens_probs[0]
        else:
            return generated_wavs, generated_texts, generated_text_tokens_probs

    def detokenize_audio(self, audio_tokens):
        # audio_tokens shape: [1, Len] as passed from loop above.
        # But if we want to change signature to support batch, we can.
        # For now, keeping signature compatible with single item usage inside loop.
        if self.detokenizer is None:
            raise ValueError("Detokenizer is not initialized")
        self.detokenizer.clear_states() # Reset for each sample
        chunk_size = 30  
        first_chunk_size = 30
        cache_speech_collection = []
        audio_tokens = audio_tokens.to(self.device).long()
        num_audio_tokens = audio_tokens.size(1)
        
        first_chunk_semantic_tokens = audio_tokens[:, :first_chunk_size]
        gen_speech = self.detokenizer.detokenize_streaming(
            first_chunk_semantic_tokens,
            is_final=(num_audio_tokens <= first_chunk_size),
            upsample_factor=4,
        )
        cache_speech_collection.append(gen_speech)

        if num_audio_tokens > first_chunk_size:
            res_semantic_tokens = audio_tokens[:, first_chunk_size:]
            for i in range(0, res_semantic_tokens.size(1), chunk_size):
                chunk_semantic_tokens = res_semantic_tokens[:, i : i + chunk_size]
                gen_speech = self.detokenizer.detokenize_streaming(
                    chunk_semantic_tokens,
                    upsample_factor=4,
                    is_final=(i + chunk_size >= res_semantic_tokens.size(1)),
                )
                cache_speech_collection.append(gen_speech)

        gen_speech = torch.cat(cache_speech_collection, dim=-1)
        return gen_speech

    def detokenize_text(self, text_tokens_list):
        # text_tokens_list: List[List[int]]
        text_batch = []
        for text_tokens in text_tokens_list:
            valid_text_ids = []
            for x in text_tokens:
                if x == self.extra_tokens.kimia_text_eos:
                    break
                valid_text_ids.append(x)
            text_batch.append(self.prompt_manager.text_tokenizer.decode(valid_text_ids))
        return text_batch

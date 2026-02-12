import os
import torch
from loguru import logger
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
        pad_token_id = self.extra_tokens.pad
        
        # Defensive: Ensure attention_mask acts correctly on padding
        # If mask is None, create it from padding.
        # If mask is provided but is all 1s (common in some infer pipelines), fix it based on padding.
        if attention_mask is None:
            if (audio_input_ids == pad_token_id).any():
                attention_mask = (audio_input_ids != pad_token_id).long()
            else:
                # No padding found or no way to know, assume all valid (ones)
                # But we create explicit mask to ensure consistent Varlen behavior
                attention_mask = torch.ones_like(audio_input_ids, dtype=torch.long)
        else:
            # Check for "Fake" mask (All 1s but content has padding)
            if attention_mask.all() and (audio_input_ids == pad_token_id).any():
                attention_mask = (audio_input_ids != pad_token_id).long()

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
        blank_text_token = torch.tensor(self.extra_tokens.kimia_text_blank, device=self.device)
        
        # Output buffers [Batch, MaxLen]
        text_previous_tokens = torch.zeros(
            (batch_size, max_new_tokens),
            dtype=torch.int,
            device=self.device,
        )
        text_previous_tokens_prob = torch.zeros(
            (batch_size, max_new_tokens),
            dtype=torch.float,
            device=self.device,
        )

        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        
        # Position IDs: [Batch, Seq]
        # NOTE (Sync with Training): Training uses standard Right Padding.
        # dataset.py: input_ids_batch.append(torch.nn.functional.pad(..., (0, audio_pad_len) ... ))
        # This confirms Right Padding is the correct alignment.
        decoder_position_ids = attention_mask.long().cumsum(-1) - 1
        decoder_position_ids.masked_fill_(attention_mask == 0, 0)
        
        batch_valid_lengths = attention_mask.sum(dim=-1).long() # [Batch]
        next_position_ids = batch_valid_lengths.unsqueeze(1).clone() # Position for NEW token
        
        decoder_input_whisper_feature = continous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None
        
        # Valid length trackers
        valid_text_length = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i in range(max_new_tokens):
            # NOTE: With Right Padding, we WILL have zeros if batch is diverse.
            forward_attention_mask = attention_mask if (attention_mask is not None and (attention_mask == 0).any()) else None

            text_logits, past_key_values = self.alm.forward(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                attention_mask=forward_attention_mask,
                use_cache=True,
                return_dict=False,
            )
            
            if decoder_input_audio_ids.shape[1] > 1:
                # First Step: Logits [B, Seq, V]
                # Gather logits at batch_valid_lengths - 1
                last_indices = (batch_valid_lengths - 1).view(-1, 1, 1).expand(-1, -1, text_logits.size(-1))
                text_logits = torch.gather(text_logits, 1, last_indices).squeeze(1)
            else:
                # Next Steps: Input is single token. Logits [B, 1, V] -> [B, V]
                if len(text_logits.shape) == 3:
                    text_logits = text_logits[:,-1,:]

            recent_text = text_previous_tokens[:, :i] if i > 0 else None
            next_token_text, max_prob_text = sampler.sample_text_logits(
                text_logits, recent_tokens=recent_text
            )
            
            # NOTE: Always use blank tokens for audio channel since we only generate text
            next_audio_token = torch.full(
                (batch_size,), 
                self.extra_tokens.kimia_text_blank, 
                dtype=torch.int, 
                device=self.device
            )

            # Update finished status
            eos_mask = (next_token_text == self.extra_tokens.kimia_text_eos)
            text_stream_is_finished = text_stream_is_finished | eos_mask
            
            # Override finished tokens with Blank/Pad
            next_token_text = torch.where(
                text_stream_is_finished, 
                blank_text_token,
                next_token_text
            )
            
            valid_text_length += (~text_stream_is_finished).long() 
            
            text_previous_tokens[:, i] = next_token_text
            text_previous_tokens_prob[:, i] = max_prob_text

            # Check global finish
            if text_stream_is_finished.all():
                break
            
            # Prepare next step
            decoder_input_audio_ids = next_audio_token.unsqueeze(1)
            decoder_input_text_ids = next_token_text.unsqueeze(1)
            
            # Update Position IDs
            decoder_position_ids = next_position_ids.clone()
            next_position_ids += 1
            
            # Update attention_mask if we were explicitly passing it (we passed it 1st step)
            # For next steps with cache, we append 1s only for active sequences, 0s for finished ones to avoid noise attention
            if attention_mask is not None:
                new_mask_col = torch.ones(batch_size, 1, device=self.device, dtype=torch.long)
                attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)
                
            decoder_input_whisper_feature = None
            decoder_is_continuous_mask = None

        # Post-Processing: Convert results to List[List]
        return_text_tokens, return_text_tokens_probs = [], []
        for b in range(batch_size):
            t_len = valid_text_length[b]
            return_text_tokens.append(
                text_previous_tokens[b, :t_len].detach().cpu().numpy().tolist()
            )
            return_text_tokens_probs.append(
                text_previous_tokens_prob[b, :t_len].detach().cpu().numpy().tolist()
            )

        return return_text_tokens, return_text_tokens_probs

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
        batch_content = self.prompt_manager.get_batch_prompt(
            chats_batch,
            output_type=output_type,
        )
        batch_content.to(self.device)
        
        audio_input_ids = batch_content.audio_input_ids
        text_input_ids = batch_content.text_input_ids
        is_continuous_mask = batch_content.is_continuous_mask
        attention_mask = batch_content.attention_mask
        audio_features = batch_content.continuous_feature # This is a flattened list of tensors

        generated_text_tokens, generated_text_tokens_probs = self._generate_loop(
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

        generated_texts = []
        # Process output batch
        for b in range(len(generated_text_tokens)):
            # Text
            text_toks = [
                t for t in generated_text_tokens[b] if t < self.kimia_token_offset
            ]
            generated_texts.append(self.detokenize_text([text_toks])[0]) # Pass list of 1 list
            
        if is_single_input:
            return generated_texts[0], generated_text_tokens_probs[0]
        else:
            return generated_texts, generated_text_tokens_probs

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
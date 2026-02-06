import torch


class KimiASampler:
    def __init__(
        self,
        audio_top_k: int,
        audio_temperature: float,
        audio_repetition_penalty: float,
        audio_repetition_window_size: int,
        text_top_k: int,
        text_temperature: float,
        text_repetition_penalty: float,
        text_repetition_window_size: int,
    ):
        self.audio_top_k = audio_top_k
        self.audio_temperature = audio_temperature
        self.text_top_k = text_top_k
        self.text_temperature = text_temperature

        self.audio_repetition_penalty = audio_repetition_penalty
        self.audio_repetition_window_size = audio_repetition_window_size
        self.text_repetition_penalty = text_repetition_penalty
        self.text_repetition_window_size = text_repetition_window_size

    def sample_audio_logits(
        self, logits: torch.Tensor, recent_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """Sample from audio logits with top-k, temperature and repetition penalty.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            recent_tokens: Optional tensor of recent tokens for repetition penalty, shape [batch_size, window_size]

        Returns:
            Sampled token ids, shape [batch_size]
        """
        # Take the last token's logits if we have a sequence dimension
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]

        # Apply repetition penalty if needed
        if (
            self.audio_repetition_penalty > 1.0
            and recent_tokens is not None
            and recent_tokens.size(1) > 0  # Check if we have any recent tokens
        ):
            # recent_tokens should be [batch_size, window_size]
            window_size = min(recent_tokens.size(1), self.audio_repetition_window_size)
            recent_window = recent_tokens[:, -window_size:].long()

            # Gather scores of recent tokens
            # logits: [batch, vocab], recent_window: [batch, window]
            # We want to select scores for tokens in recent_window
            scores = torch.gather(logits, dim=1, index=recent_window)

            # Apply penalty: if score < 0 multiply by penalty, otherwise divide by penalty
            scores = torch.where(
                scores < 0,
                scores * self.audio_repetition_penalty,
                scores / self.audio_repetition_penalty,
            )

            # Put the penalized scores back
            logits.scatter_(dim=1, index=recent_window, src=scores)

        # Convert to probabilities with softmax
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # Apply temperature scaling if not greedy
        if self.audio_temperature > 1e-6:
            logprobs = logprobs / self.audio_temperature
            probs = torch.exp(logprobs)

            # Apply top-k sampling
            if self.audio_top_k > 0:
                # Select top-k probabilities and indices
                top_k_probs, top_k_indices = torch.topk(probs, self.audio_top_k, dim=-1)

                # Sample from the top-k distribution
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
                
                # Gather original indices
                next_token = torch.gather(top_k_indices, -1, sampled_indices).squeeze(-1)
            else:
                # Sample from the full distribution
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            # Greedy sampling (temperature = 0)
            next_token = torch.argmax(logprobs, dim=-1)

        return next_token

    def sample_text_logits(
        self, logits: torch.Tensor, recent_tokens: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from text logits with top-k, temperature and repetition penalty.
        
        Returns:
            next_token: [batch_size]
            max_prob: [batch_size]
        """
        # Take the last token's logits if we have a sequence dimension
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]

        if (
            self.text_repetition_penalty > 1.0
            and recent_tokens is not None
            and recent_tokens.size(1) > 0
        ):
            window_size = min(recent_tokens.size(1), self.text_repetition_window_size)
            recent_window = recent_tokens[:, -window_size:].long()

            scores = torch.gather(logits, dim=1, index=recent_window)
            scores = torch.where(
                scores < 0,
                scores * self.text_repetition_penalty,
                scores / self.text_repetition_penalty,
            )
            logits.scatter_(dim=1, index=recent_window, src=scores)

        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        if self.text_temperature > 1e-6:
            logprobs = logprobs / self.text_temperature
            probs = torch.exp(logprobs)

            if self.text_top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, self.text_top_k, dim=-1)
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
                next_token = torch.gather(top_k_indices, -1, sampled_indices).squeeze(-1)
            else:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # For probability return, we gather the prob of the selected token from original logprobs
            max_prob = torch.gather(torch.exp(logprobs), 1, next_token.unsqueeze(1)).squeeze(1)
        else:
            next_token = torch.argmax(logprobs, dim=-1)
            # greedy decoding, return the probability of the selected token
            max_prob = torch.gather(torch.exp(logprobs), 1, next_token.unsqueeze(1)).squeeze(1)

        return next_token, max_prob
        """Sample from text logits with top-k, temperature and repetition penalty.

        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
            recent_tokens: Optional tensor of recent tokens for repetition penalty

        Returns:
            Sampled token ids
        """
        # Take the last token's logits if we have a sequence dimension
        if len(logits.shape) == 3:
            logits = logits[:, -1]

        # Apply repetition penalty if needed
        if (
            self.text_repetition_penalty > 1.0
            and recent_tokens is not None
            and len(recent_tokens) > self.text_repetition_window_size
        ):
            logits = logits[0]  # Assumes batch size of 1 for repetition penalty
            recent_window = recent_tokens[-self.text_repetition_window_size :].long()

            # Gather scores of recent tokens
            scores = torch.gather(logits, dim=0, index=recent_window)

            # Apply penalty: if score < 0 multiply by penalty, otherwise divide by penalty
            scores = torch.where(
                scores < 0,
                scores * self.text_repetition_penalty,
                scores / self.text_repetition_penalty,
            )

            # Put the penalized scores back
            logits.scatter_(dim=0, index=recent_window, src=scores)
            logits = logits.unsqueeze(0)  # Add batch dimension back

        # Convert to probabilities with softmax
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        # Apply temperature scaling if not greedy
        if self.text_temperature > 1e-6:
            logprobs = logprobs / self.text_temperature

            # Apply top-k sampling
            if self.text_top_k > 0:
                # Get probabilities from logprobs
                probs = torch.exp(logprobs)

                # Select top-k probabilities and indices
                top_k_probs, top_k_indices = torch.topk(probs, self.text_top_k, dim=-1)

                # Sample from the top-k distribution
                sampled_indices = torch.multinomial(top_k_probs, num_samples=1).squeeze(
                    1
                )
                next_token = top_k_indices.gather(
                    -1, sampled_indices.unsqueeze(-1)
                ).squeeze(-1)
            else:
                # Sample from the full distribution
                next_token = torch.multinomial(
                    torch.exp(logprobs), num_samples=1
                ).squeeze(1)
        else:
            # Greedy sampling (temperature = 0)
            # print('Greedy sampling activated')
            # print('logprobs:', logprobs.shape)
            next_token = torch.argmax(logprobs, dim=-1)
            # print('next_token:', next_token)

            # 同时获取最大值和索引
            max_values, _ = torch.max(logprobs, dim=-1)
            max_probs = torch.exp(max_values) 
            # print('max_probability:', max_probs)
            # print('next_token:', next_token)
        return next_token, max_probs
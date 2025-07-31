# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that samples the next tokens from the model's outputs."""

import torch
import torch.nn as nn

from vllm.config import LogprobsMode
from vllm.utils import is_pin_memory_available
from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.penalties import apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.logger import init_logger

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):
    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs"):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()
        self.pin_memory = is_pin_memory_available()
        self.logprobs_mode = logprobs_mode

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        # TODO(rob): provide option for logprobs post sampling.
        # See https://vllm-dev.slack.com/archives/C07UUL8E61Z/p1735907856007919 # noqa: E501
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if self.logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif self.logprobs_mode == "raw_logits":
                raw_logprobs = logits.clone()

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Apply allowed token ids.
        logits = self.apply_allowed_token_ids(logits, sampling_metadata)
        # Apply bad words exclusion.
        logits = self.apply_bad_words(logits, sampling_metadata)

        # Apply logits processors which can impact greedy sampling
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)

        # Apply penalties (e.g., min_tokens, freq_penalties).
        logits = self.apply_penalties(logits, sampling_metadata)

        # Get the process logprobs or logits.
        if num_logprobs is not None:
            if self.logprobs_mode == "processed_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif self.logprobs_mode == "processed_logits":
                raw_logprobs = logits.clone()

        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        # Gather the logprobs of the topk and sampled token (if requested).
        # Get logprobs and rank tensors (if requested)
        logprobs_tensors = (
            None
            if num_logprobs is None
            else self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )
        )

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    # Dynamic Temperature
    def compute_dynamic_temperature(
        self,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """
        Compute dynamic temperature based on generation step.

        Temperature decreases linearly from initial_temperature to final_temperature
        over the course of generation steps.

        Args:
            sampling_metadata: Contains dynamic temperature parameters and current step info

        Returns:
            torch.Tensor: Updated temperature values for each request
        """
        # Start with the base temperature
        temperature = sampling_metadata.temperature

        # Check individual conditions and log warnings
        has_use_dynamic_temperature = hasattr(
            sampling_metadata, "use_dynamic_temperature"
        )
        if not has_use_dynamic_temperature:
            logger.warning(
                "Dynamic temperature requested but 'use_dynamic_temperature' attribute not found in sampling_metadata"
            )
            return temperature

        use_dynamic_temperature_enabled = has_use_dynamic_temperature and (
            sampling_metadata.use_dynamic_temperature.any()
            if torch.is_tensor(sampling_metadata.use_dynamic_temperature)
            else sampling_metadata.use_dynamic_temperature
        )
        if has_use_dynamic_temperature and not use_dynamic_temperature_enabled:
            # logger.warning("Dynamic temperature disabled via use_dynamic_temperature=False")
            return temperature

        has_initial_temperature = hasattr(
            sampling_metadata, "initial_temperature"
        )
        if use_dynamic_temperature_enabled and not has_initial_temperature:
            logger.warning(
                "Dynamic temperature enabled but 'initial_temperature' attribute not found in sampling_metadata"
            )

        has_final_temperature = hasattr(sampling_metadata, "final_temperature")
        if use_dynamic_temperature_enabled and not has_final_temperature:
            logger.warning(
                "Dynamic temperature enabled but 'final_temperature' attribute not found in sampling_metadata"
            )

        has_current_step = hasattr(sampling_metadata, "current_step")
        if use_dynamic_temperature_enabled and not has_current_step:
            logger.warning(
                "Dynamic temperature enabled but 'current_step' attribute not found in sampling_metadata"
            )

        has_max_steps = hasattr(sampling_metadata, "max_steps")
        if use_dynamic_temperature_enabled and not has_max_steps:
            logger.warning(
                "Dynamic temperature enabled but 'max_steps' attribute not found in sampling_metadata"
            )

        if (
            has_use_dynamic_temperature
            and use_dynamic_temperature_enabled
            and has_initial_temperature
            and has_final_temperature
            and has_current_step
            and has_max_steps
        ):
            # Get dynamic temperature parameters
            initial_temp = sampling_metadata.initial_temperature
            final_temp = sampling_metadata.final_temperature
            # Use current_step + 1 because we want the temperature for the NEXT token
            # current_step represents completed steps, but we're generating step current_step + 1
            current_step = sampling_metadata.current_step + 1
            logger.warning(f"current step: {current_step}")
            max_steps = sampling_metadata.max_steps

            # Compute linear interpolation factor
            # Clamp to avoid division by zero and ensure valid range
            step_ratio = torch.clamp(
                current_step.float() / (max_steps.float() - 1), 0.0, 1.0
            )

            # Handle case where max_steps <= 1
            step_ratio = torch.where(
                max_steps <= 1, torch.ones_like(step_ratio), step_ratio
            )

            # Linear interpolation from initial to final temperature
            dynamic_temp = (
                initial_temp + (final_temp - initial_temp) * step_ratio
            )
            logger.warning(f"dynamic temp: {dynamic_temp}")

            # Apply dynamic temperature only to requests that have it enabled
            if torch.is_tensor(sampling_metadata.use_dynamic_temperature):
                # Per-request dynamic temperature control
                mask = sampling_metadata.use_dynamic_temperature
                temperature = torch.where(mask, dynamic_temp, temperature)
            else:
                # Apply to all requests
                temperature.copy_(dynamic_temp)

        # logger.warning(f"temperature: {temperature}")
        return temperature

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        # logger.warning(f"Applying temperature {temp}")
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    # XTC
    def apply_xtc(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Apply XTC (Exclude Top Choice) filtering to logits."""
        if (
            sampling_metadata.use_xtc is None
            or not sampling_metadata.use_xtc.any()
        ):
            return logits

        for i in range(logits.shape[0]):
            if not sampling_metadata.use_xtc[i]:
                continue

            exclude_top = sampling_metadata.xtc_exclude_top[i].item()
            exclusion_threshold = sampling_metadata.xtc_exclusion_threshold[
                i
            ].item()
            min_probability = sampling_metadata.xtc_min_probability[i].item()

            # Get top-k logits and indices directly
            top_logits, top_indices = torch.topk(
                logits[i], k=min(exclude_top + 10, logits.shape[-1])
            )

            # Track discarded token IDs for warning
            discarded_ids = []

            # We need to convert into probabilities to use the threshold but at least now I compute the softmax outside the inner loop, which was dumb
            probs_i = torch.softmax(logits[i], dim=-1)

            for j in range(min(exclude_top, len(top_indices))):
                top_idx = top_indices[j].item()
                top_prob = probs_i[top_idx].item()

                # Exclude if probability exceeds threshold and is above minimum
                if (
                    top_prob >= exclusion_threshold
                    and top_prob >= min_probability
                ):
                    logits[i, top_idx] = float("-inf")
                    discarded_ids.append(top_idx)

            if discarded_ids:
                logger.warning(f"XTC discarded token IDs: {discarded_ids}")
        return logits

    # DRY
    def apply_dry_penalty(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Apply DRY (Don't Repeat Yourself) penalty to prevent sequence looping."""
        if (
            sampling_metadata.use_dry is None
            or not sampling_metadata.use_dry.any()
        ):
            # logger.warning("DRY DEBUG: Early return - use_dry conditions not met")
            return logits

        # Check if we have any context at all (either prompt or output tokens)
        has_context = False
        if sampling_metadata.prompt_token_ids is not None:
            has_context = True
        elif hasattr(sampling_metadata, 'output_token_ids') and sampling_metadata.output_token_ids:
            has_context = True
        
        if not has_context:
            # logger.warning("DRY DEBUG: No context available (no prompt_token_ids or output_token_ids)")
            return logits

        # logger.warning("DRY DEBUG: Proceeding with DRY penalty application")

        for i in range(logits.shape[0]):
            if not sampling_metadata.use_dry[i]:
                continue

            # Get parameters for this request
            multiplier = sampling_metadata.dry_multiplier[i].item()
            base = sampling_metadata.dry_base[i].item()
            allowed_length = sampling_metadata.dry_allowed_length[i].item()
            sequence_breakers = set(sampling_metadata.dry_sequence_breakers.get(i, []))

            # logger.warning(f"DRY DEBUG: Request {i} - multiplier={multiplier}, base={base}, allowed_length={allowed_length}")

            # Get the full context - handle case where prompt_token_ids might be None
            full_context = []
            
            # Add prompt tokens if available
            if sampling_metadata.prompt_token_ids is not None and i < len(sampling_metadata.prompt_token_ids):
                prompt_tokens = sampling_metadata.prompt_token_ids[i]
                if hasattr(prompt_tokens, 'tolist'):
                    full_context.extend(prompt_tokens.tolist())
                elif isinstance(prompt_tokens, list):
                    full_context.extend(prompt_tokens)
            
            # Add output tokens if available
            if hasattr(sampling_metadata, 'output_token_ids') and sampling_metadata.output_token_ids:
                if i < len(sampling_metadata.output_token_ids):
                    output_tokens = sampling_metadata.output_token_ids[i]
                    if isinstance(output_tokens, list):
                        full_context.extend(output_tokens)
                    elif hasattr(output_tokens, 'tolist'):
                        full_context.extend(output_tokens.tolist())

            # logger.warning(f"DRY DEBUG: Full context length: {len(full_context)}")
            # logger.warning(f"DRY DEBUG: Last 20 tokens: {full_context[-20:]}")
            
            # Skip if context is too short
            if len(full_context) < allowed_length + 1:  # Need at least allowed_length + 1 for any penalty
                # logger.warning(f"DRY DEBUG: Context too short ({len(full_context)} < {allowed_length + 1})")
                continue

            # Pre-compute candidate tokens to check
            candidate_tokens = self._get_dry_candidate_tokens(full_context, sequence_breakers)
            
            # logger.warning(f"DRY DEBUG: Found {len(candidate_tokens)} candidate tokens: {sorted(list(candidate_tokens))}")
            
            penalties_applied = 0
            total_penalty = 0.0

            # Check each candidate token
            for token_id in candidate_tokens:
                penalty = self._calculate_dry_penalty(
                    full_context,
                    token_id,
                    multiplier,
                    base,
                    allowed_length,
                    sequence_breakers,
                )
                if penalty > 0:
                    logits[i, token_id] -= penalty
                    penalties_applied += 1
                    total_penalty += penalty

            logger.warning(f"DRY DEBUG: Applied {penalties_applied} penalties, total_penalty={total_penalty:.3f}")

        return logits


    def _get_dry_candidate_tokens(self, context, sequence_breakers):
        """Get tokens that could potentially extend matching sequences (optimization)."""
        if len(context) < 2:
            return set()
        
        # Look at recent tokens that could be part of repeated sequences
        # This avoids checking every token in the vocabulary
        recent_window = min(200, len(context))
        candidate_tokens = set(context[-recent_window:])
        
        # Also include tokens that appear right after sequence breakers
        # as these are common repetition points
        for i in range(len(context) - 1):
            if context[i] in sequence_breakers and i + 1 < len(context):
                candidate_tokens.add(context[i + 1])
        
        return candidate_tokens

    def _calculate_dry_penalty(
        self,
        context,
        next_token,
        multiplier,
        base,
        allowed_length,
        sequence_breakers,
    ) -> float:
        """Calculate DRY penalty for a specific token."""
        if len(context) < allowed_length:
            return 0.0

        # What the context would look like with next_token added
        extended_context = context + [next_token]
        
        # logger.warning(f"DRY DEBUG: Checking token {next_token}, context_len={len(context)}, allowed_length={allowed_length}")
        
        # Find the longest sequence ending at the current position that 
        # matches a sequence appearing earlier in the context
        max_match_length = self._find_longest_dry_repetition(
            extended_context, sequence_breakers
        )
        
        # logger.warning(f"DRY DEBUG: Found max_match_length={max_match_length} for token {next_token}")

        # Apply penalty if match exceeds allowed length
        if max_match_length > allowed_length:
            penalty = multiplier * (base ** (max_match_length - allowed_length))
            # logger.warning(f"DRY PENALTY APPLIED: match_length={max_match_length}, penalty={penalty:.3f}, token={next_token}")
            return penalty
        # logger.warning(f"DRY NO PENALTY: match_length={max_match_length} <= allowed_length={allowed_length}")
        return 0.0

    def _find_longest_dry_repetition(self, context, sequence_breakers):
        """Find the longest sequence at the end that repeats from earlier in context."""
        context_len = len(context)
        max_match_length = 0
        
        # Start from longest possible and work down (optimization)
        max_possible_length = min(context_len // 2, context_len)
        
        for seq_len in range(max_possible_length, 0, -1):
            if seq_len <= max_match_length:
                break  # Already found a longer match
                
            # The sequence we're checking (at the end)
            end_sequence = context[-seq_len:]
            
            # Skip if this sequence contains breakers - they reset matching
            if any(token in sequence_breakers for token in end_sequence):
                continue
            
            # Look for this exact sequence earlier in the context
            # Stop before we overlap with the end sequence
            search_limit = context_len - seq_len
            
            for start_pos in range(search_limit):
                candidate_sequence = context[start_pos:start_pos + seq_len]
                
                # Skip if candidate contains breakers
                if any(token in sequence_breakers for token in candidate_sequence):
                    continue
                
                # Check for exact match
                if candidate_sequence == end_sequence:
                    max_match_length = max(max_match_length, seq_len)
                    break  # Found match for this length, try longer
        
        return max_match_length

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        assert not (
            sampling_metadata.all_greedy and sampling_metadata.all_random
        )
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        assert sampling_metadata.temperature is not None

        dynamic_temperature = self.compute_dynamic_temperature(
            sampling_metadata
        )
        # Apply temperature.
        logits = self.apply_temperature(logits, dynamic_temperature)

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        logits = self.apply_xtc(logits, sampling_metadata)
        logits = self.apply_dry_penalty(logits, sampling_metadata)
        # Apply top_k and/or top_p.
        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_penalties(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if not sampling_metadata.no_penalties:
            assert sampling_metadata.prompt_token_ids is not None
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
                sampling_metadata.output_token_ids,
            )
        return logits

    def apply_allowed_token_ids(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(
                sampling_metadata.allowed_token_ids_mask, float("-inf")
            )
        return logits

    def apply_bad_words(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        if sampling_metadata.bad_words_token_ids:
            apply_bad_words(
                logits,
                sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids,
            )
        return logits

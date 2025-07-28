import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.v1.sample.ops.topk_topp_sampler import random_sample


class DynamicTemperatureSampler(nn.Module):
    def __init__(
        self,
        initial_temp: float = 1.2,
        final_temp: float = 0.6,
        decay_steps: int = 50,
    ):
        super().__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.decay_steps = decay_steps

    def _get_temperature(self, step: int) -> float:
        if step >= self.decay_steps:
            return self.final_temp
        return self.initial_temp - (self.initial_temp - self.final_temp) * (
            step / self.decay_steps
        )

    def forward(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        step: int,
    ):
        current_temp = self._get_temperature(step)
        probs = F.softmax(logits / current_temp, dim=-1)
        return random_sample(probs, generators)

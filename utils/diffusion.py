import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


class TtoSigma:
    def __init__(self, tr_sigma_min: float, tr_sigma_max: float,
                 rot_sigma_min: float, rot_sigma_max: float,
                 tor_sigma_min: float, tor_sigma_max: float):
        self.tr_sigma_min = tr_sigma_min
        self.tr_sigma_max = tr_sigma_max
        self.rot_sigma_min = rot_sigma_min
        self.rot_sigma_max = rot_sigma_max
        self.tor_sigma_min = tor_sigma_min
        self.tor_sigma_max = tor_sigma_max

    def __call__(self,
                 t_tr: Union[float, torch.Tensor],
                 t_rot: Union[float, torch.Tensor],
                 t_tor: Union[float, torch.Tensor]):
        tr_sigma = self.tr_sigma_min ** (1 - t_tr) * self.tr_sigma_max ** t_tr
        rot_sigma = self.rot_sigma_min ** (1 - t_rot) * self.rot_sigma_max ** t_rot
        tor_sigma = self.tor_sigma_min ** (1 - t_tor) * self.tor_sigma_max ** t_tor
        return tr_sigma, rot_sigma, tor_sigma


class SinusoidalEmbedding:
    def __init__(self, embedding_dim: int, embedding_scale: int, max_positions: int = 10_000):
        self.embedding_dim = embedding_dim
        self.embedding_scale = embedding_scale
        self.max_positions = max_positions

    def __call__(self, timesteps):
        """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
        timesteps = timesteps * self.embedding_scale
        assert len(timesteps.shape) == 1
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], self.embedding_dim)
        return emb


def get_t_schedule(inference_steps):
    return np.linspace(1, 0, inference_steps + 1)[:-1]

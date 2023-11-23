import os
from pathlib import Path

import numpy as np

from .general import logger

"""
    Preprocessing for the SO(2)/torus sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the pre-computation is only run the first time the repository is run on a machine
"""

cache_path = Path("cache/")
np.seterr(divide='ignore', invalid='ignore')


def p(x, sigma, n=10):
    _p = 0
    for i in range(-n, n + 1):
        _p += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return _p


def grad(x, sigma, n=10):
    _p = 0
    for i in range(-n, n + 1):
        _p += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return _p


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x_ = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma_ = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

if (cache_path / '.p.npy').exists():
    p_ = np.load(str(cache_path / '.p.npy'))
    score_ = np.load(str(cache_path / '.score.npy'))
else:
    os.makedirs(cache_path, exist_ok=True)
    logger.info(f"precomputing and saving torus distribution table to cache {str(cache_path.absolute())}")
    p_ = p(x_, sigma_[:, None], n=100)
    np.save(str(cache_path / '.p.npy'), p_)
    score_ = grad(x_, sigma_[:, None], n=100) / p_
    np.save(str(cache_path / '.score.npy'), score_)


def score(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


score_norm_ = score(
    sample(sigma_[None].repeat(10000, 0).flatten()),
    sigma_[None].repeat(10000, 0).flatten()
).reshape(10000, -1)
score_norm_ = (score_norm_ ** 2).mean(0)


def score_norm(sigma):
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]

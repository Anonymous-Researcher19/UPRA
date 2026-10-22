from __future__ import annotations

import numpy as np


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.clip(np.linalg.norm(left, axis=-1, keepdims=True), 1e-8, None)
    right = right / np.clip(np.linalg.norm(right, axis=-1, keepdims=True), 1e-8, None)
    return np.sum(left * right, axis=-1)


def preference_similarity(generated: np.ndarray, histories: np.ndarray) -> np.ndarray:
    """CLAP-P/PaSST-P: generated embedding vs. mean user-history embedding."""
    return cosine_rows(generated, histories.mean(axis=1))


def kl_divergence(generated_posterior: np.ndarray, history_posteriors: np.ndarray) -> np.ndarray:
    mean_history = history_posteriors.mean(axis=1)
    p = np.clip(generated_posterior, 1e-8, 1.0)
    q = np.clip(mean_history, 1e-8, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def frechet_distance(generated: np.ndarray, targets: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    mu_g, mu_t = generated.mean(0), targets.mean(0)
    cov_g, cov_t = np.cov(generated, rowvar=False), np.cov(targets, rowvar=False)
    root = sqrtm(cov_g @ cov_t)
    if np.iscomplexobj(root):
        root = root.real
    return float(np.sum((mu_g - mu_t) ** 2) + np.trace(cov_g + cov_t - 2 * root))


def audiobox_score(production_quality: np.ndarray, content_enjoyment: np.ndarray, content_usefulness: np.ndarray) -> np.ndarray:
    """Paper Eq. (18); Production Complexity is intentionally excluded."""
    return (production_quality + content_enjoyment + content_usefulness) / 3.0


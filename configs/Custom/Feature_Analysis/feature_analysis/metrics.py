# -*- coding: utf-8 -*-
"""Representation-comparison and dimensionality metrics.

All functions operate on NumPy arrays and are free of GPU / model
dependencies, so the analysis phase can run on CPU from cached features.

Metric summary
--------------
debiased_linear_cka : unbiased linear centered kernel alignment between two
    location-matched representations (Song et al., 2012; Nguyen et al., 2021).
debiased_rbf_cka    : non-linear counterpart with a median-heuristic bandwidth.
bures_fidelity      : trace-normalised Bures fidelity (Gaussian fidelity /
    2-Wasserstein geometry) between two channel covariances; the headline
    resolution-invariance descriptor -- scale-invariant and registration-free.
effective_rank      : exp(entropy of normalised eigenvalues) of a covariance.
participation_ratio : (Sum lambda)^2 / Sum lambda^2.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Centered kernel alignment (location-matched representations)
# --------------------------------------------------------------------------- #
def _gram(x: np.ndarray) -> np.ndarray:
    return x @ x.T


def _unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased Hilbert-Schmidt independence criterion (Song et al., 2012)."""
    n = K.shape[0]
    K = K.copy(); L = L.copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    ones = np.ones(n)
    term1 = np.sum(K * L)
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * (ones @ K @ L @ ones)
    return float((term1 + term2 - term3) / (n * (n - 3)))


def debiased_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased linear CKA. X:(n,p), Y:(n,q); rows are matched samples."""
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Row counts (matched locations) must agree for CKA.")
    if X.shape[0] < 4:
        return float("nan")
    Kx, Ky = _gram(X), _gram(Y)
    hxy = _unbiased_hsic(Kx, Ky)
    hxx = _unbiased_hsic(Kx, Kx)
    hyy = _unbiased_hsic(Ky, Ky)
    denom = np.sqrt(max(hxx, 0.0) * max(hyy, 0.0))
    return float(hxy / denom) if denom > 0 else float("nan")


def _rbf_gram(x: np.ndarray) -> np.ndarray:
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    sigma2 = med if med > 0 else 1.0
    return np.exp(-d2 / (2.0 * sigma2))


def debiased_rbf_cka(X: np.ndarray, Y: np.ndarray) -> float:
    if X.shape[0] != Y.shape[0] or X.shape[0] < 4:
        return float("nan")
    Kx, Ky = _rbf_gram(X), _rbf_gram(Y)
    hxy = _unbiased_hsic(Kx, Ky)
    hxx = _unbiased_hsic(Kx, Kx)
    hyy = _unbiased_hsic(Ky, Ky)
    denom = np.sqrt(max(hxx, 0.0) * max(hyy, 0.0))
    return float(hxy / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Second-order channel statistics (registration-free, cross-resolution)
# --------------------------------------------------------------------------- #
def channel_covariance(rows: np.ndarray) -> np.ndarray:
    """Channel covariance (d x d) from a (N, d) sample matrix."""
    rows = rows - rows.mean(axis=0, keepdims=True)
    n = max(rows.shape[0] - 1, 1)
    return (rows.T @ rows) / n


def covariance_cosine(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius cosine of two covariance matrices (secondary descriptor;
    dominated by the diagonal variance profile, hence a high floor)."""
    num = float(np.sum(A * B))
    den = float(np.linalg.norm(A) * np.linalg.norm(B))
    return num / den if den > 0 else float("nan")


def bures_fidelity(A: np.ndarray, B: np.ndarray, eps: float = 1e-6) -> float:
    """Trace-normalised Bures fidelity in [0, 1] between two covariances.

    Equivalent to the quantum fidelity between the corresponding zero-mean
    Gaussian feature distributions after trace normalisation; a scale-invariant
    similarity derived from the 2-Wasserstein (Bures) geometry on symmetric
    positive-definite matrices. Equals 1 iff the trace-normalised covariances
    coincide, and compares the full second-order structure rather than the
    diagonal alone.
    """
    d = A.shape[0]
    ta, tb = np.trace(A), np.trace(B)
    if ta <= 0 or tb <= 0:
        return float("nan")
    A = A / ta + eps * np.eye(d) / d
    B = B / tb + eps * np.eye(d) / d
    wa, Ua = np.linalg.eigh(A)
    wa = np.clip(wa, 0.0, None)
    A_half = (Ua * np.sqrt(wa)) @ Ua.T
    M = A_half @ B @ A_half
    wm = np.clip(np.linalg.eigvalsh(M), 0.0, None)
    return float(np.sum(np.sqrt(wm)) ** 2)


# --------------------------------------------------------------------------- #
# Representational dimensionality
# --------------------------------------------------------------------------- #
def effective_rank(cov: np.ndarray) -> float:
    """exp(entropy of normalised eigenvalues) -- continuous dimensionality."""
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 1e-12]
    if ev.size == 0:
        return 0.0
    p = ev / ev.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio(cov: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 0]
    if ev.size == 0:
        return 0.0
    return float((ev.sum() ** 2) / np.sum(ev ** 2))

# core_experiment.py
# --------------------------------------------------------------------------------------
# Single source of truth for your experiments:
#   • build_five_kernels(fs, rng): returns the 5 kernels (with comments & parameters)
#   • generate_forcing(N, fs, rng): common forcing generator
#   • prony_from_log_rates(...): approximates stretched-exponential & power-law kernels
#   • hash_array_sha256(x): robust reproducibility check for forcing
#   • make_manifest(path, ...): writes all params/kernels/seeds + forcing hash to JSON
#
# Both numerical solvers and ML pipelines should import from this file so they share
# *exactly* the same kernels, forcing process, and parameters.
# --------------------------------------------------------------------------------------

from __future__ import annotations
import json, hashlib, math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np


# ---------------------------
# Small dataclass for kernels
# ---------------------------
@dataclass
class KernelSpec:
    """Holds a single kernel as a Prony (generalized Maxwell) sum: K(t)=Σ a_m e^{-b_m t}."""
    name: str
    a: np.ndarray           # nonnegative amplitudes (shape [M])
    b: np.ndarray           # positive decay rates (shape [M])
    # Extra metadata for documentation/debug (e.g., target beta, tau, alpha)
    meta: Dict[str, Any]


# -------------------------------------------------------
# Prony approximation for stretched exp & power-law K(t)
# -------------------------------------------------------
def prony_from_log_rates(
    beta_like: float,
    A: float,
    tau: float | None,
    M: int,
    tmin: float,
    tmax: float,
    kind: str = "stretched",
    rng: np.random.Generator | None = None,
    iters: int = 2000,
    lr: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a Prony series K(t) ≈ Σ a_m e^{-b_m t} that approximates:
      • kind='stretched': K_true(t)=A * exp(-(t/tau)^beta),           0<beta<1
      • kind='power':    K_true(t)≈A * t^{-beta} over [tmin, tmax],   0<beta<1

    Strategy:
      1) Pick M logarithmically spaced decay rates b_m ∈ [1/tmax, 1/tmin].
      2) Fit nonnegative weights a_m via projected gradient NNLS on a log-spaced time grid.
    """
    assert kind in ("stretched", "power")
    # Decay rates that cover the desired time window:
    b_min = 1.0 / (tmax + 1e-12)
    b_max = 1.0 / (max(tmin, 1e-6))
    b_vec = np.logspace(np.log10(b_min), np.log10(b_max), M)

    # Fit on a log time grid to capture both early/late behavior
    Tfit = 512
    t_grid = np.geomspace(max(tmin, 1e-6), tmax, Tfit)

    # Target function values
    if kind == "stretched":
        if tau is None:
            raise ValueError("tau must be provided for 'stretched' approximation.")
        target = A * np.exp(-np.power(t_grid / tau, beta_like))
    else:  # power-law ~ A * t^{-beta}
        target = A * np.power(np.maximum(t_grid, 1e-6), -beta_like)

    # Design matrix: Phi_{i,m} = e^{-b_m * t_i}
    Phi = np.exp(-np.outer(t_grid, b_vec))

    # Simple projected gradient NNLS (keeps a_m >= 0, avoids SciPy dependency)
    a = np.ones(M) * (target.max() / max(M, 1))
    for _ in range(iters):
        grad = Phi.T @ (Phi @ a - target) / Tfit
        a -= lr * grad
        a = np.maximum(a, 0.0)

    # Tiny jitter to avoid degeneracy
    if rng is not None:
        a *= (1.0 + 0.01 * rng.standard_normal(a.shape))
        a = np.maximum(a, 0.0)

    return a, b_vec


# -------------------------------------------
# Five kernels (as requested) with clear docs
# -------------------------------------------
def build_five_kernels(fs: float, rng: np.random.Generator) -> List[KernelSpec]:
    """
    Returns:
      List[KernelSpec] for the five kernel families:

      1) Maxwell single-exponential:
           K(t) = a * exp(-b t)
         • single relaxation time; very common in simple viscoelastic models.

      2) Bi-exponential (Prony, M=2):
           K(t) = a1 * exp(-b1 t) + a2 * exp(-b2 t)
         • two distinct time scales capture fast/slow damping.

      3) Tri-exponential (Prony, M=3):
           K(t) = Σ_{m=1..3} a_m * exp(-b_m t)
         • richer relaxation spectrum (common in polymers).

      4) Stretched-exponential (Kohlrausch) APPROXIMATED by Prony:
           True:  K(t) = A * exp(-(t / tau)^beta), 0 < beta < 1
           Approx: Σ a_m e^{-b_m t} fitted over [tmin, tmax]
         • models broad spectrum relaxation (glassy polymers, complex materials).

      5) Power-law kernel APPROXIMATED by Prony:
           True:  K(t) ~ c * t^{-alpha} / Γ(1-α), 0 < α < 1
           Approx: Σ a_m e^{-b_m t} fitted over [tmin, tmax]
         • long-memory behavior with heavy tails on finite windows.
    """
    dt = 1.0 / fs
    # Fitting window for non-Prony targets; excludes t≈0 singularities and covers long memory
    tmin, tmax = 5 * dt, 10.0

    kernels: List[KernelSpec] = []

    # (1) Maxwell single-exponential
    a1 = rng.uniform(0.6, 1.2)      # amplitude (nonnegative)
    b1 = rng.uniform(0.8, 2.0)      # rate [s^-1] ~ time constants 0.5..1.25 s
    kernels.append(KernelSpec(
        name="maxwell_single",
        a=np.array([a1], dtype=float),
        b=np.array([b1], dtype=float),
        meta={"family": "single_exp"}))

    # (2) Bi-exponential: two relaxation scales (fast + slow)
    a2 = rng.uniform(0.2, 0.8, size=2)
    b2 = np.sort(rng.uniform(0.2, 5.0, size=2))  # slower..faster
    kernels.append(KernelSpec(
        name="prony_biexp",
        a=a2.astype(float),
        b=b2.astype(float),
        meta={"family": "prony", "M": 2}))

    # (3) Tri-exponential: richer spectrum
    a3 = rng.uniform(0.1, 0.6, size=3)
    b3 = np.sort(rng.uniform(0.1, 7.0, size=3))
    kernels.append(KernelSpec(
        name="prony_triexp",
        a=a3.astype(float),
        b=b3.astype(float),
        meta={"family": "prony", "M": 3}))

    # (4) Stretched-exponential (Kohlrausch) approximated by Prony
    beta = rng.uniform(0.45, 0.85)    # 0<beta<1 controls "stretch"
    A = rng.uniform(0.5, 1.2)         # amplitude scaling
    tau = rng.uniform(0.2, 2.0)       # characteristic time [s]
    a4, b4 = prony_from_log_rates(
        beta_like=beta, A=A, tau=tau, M=10, tmin=tmin, tmax=tmax,
        kind="stretched", rng=rng
    )
    kernels.append(KernelSpec(
        name="stretched_exp_approx",
        a=a4.astype(float),
        b=b4.astype(float),
        meta={"family": "stretched_exp", "beta": float(beta), "A": float(A), "tau": float(tau)}))

    # (5) Power-law kernel approximated by Prony
    alpha = rng.uniform(0.3, 0.7)     # α in (0,1) → long memory
    A5 = rng.uniform(0.1, 0.4)        # amplitude
    a5, b5 = prony_from_log_rates(
        beta_like=alpha, A=A5, tau=None, M=12, tmin=tmin, tmax=tmax,
        kind="power", rng=rng
    )
    kernels.append(KernelSpec(
        name="powerlaw_approx",
        a=a5.astype(float),
        b=b5.astype(float),
        meta={"family": "power_law", "alpha": float(alpha), "A": float(A5), "fit_window": [float(tmin), float(tmax)]}))

    return kernels


# ----------------------------------------
# Common forcing for *all* kernels (shared)
# ----------------------------------------
def generate_forcing(N: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """
    Composite forcing f[n] that mimics lab excitations and gives rich dynamics:
      • colored noise (IIR low-pass of white noise),
      • sporadic short sine bursts,
      • a single medium chirp.

    IMPORTANT: Call this ONCE and use the same f for all kernels and solvers to
               guarantee apples-to-apples comparisons.
    """
    f = np.zeros(N, dtype=np.float64)

    # (1) colored (low-passed) noise
    wn = rng.standard_normal(N)
    alpha = 0.03  # smaller => smoother (more low-frequency content)
    for n in range(1, N):
        wn[n] = (1 - alpha) * wn[n - 1] + alpha * wn[n]
    f += 0.2 * wn

    # (2) a few short sine bursts (transient resonant kicks)
    n_bursts = 5
    burst_len = int(0.05 * fs)  # 50 ms
    for _ in range(n_bursts):
        start = rng.integers(0, max(1, N - burst_len))
        freq = rng.uniform(1.0, 20.0)   # Hz
        amp  = rng.uniform(0.5, 1.5)
        phi  = rng.uniform(0, 2 * np.pi)
        n = np.arange(burst_len)
        seg = amp * np.sin(2 * np.pi * freq * (n / fs) + phi)
        f[start:start + burst_len] += seg

    # (3) a medium chirp (slow→fast)
    chirp_len = int(2.0 * fs)  # 2 s
    start = rng.integers(0, max(1, N - chirp_len))
    f0, f1 = 0.5, 40.0  # Hz sweep
    amp = 1.0
    n = np.arange(chirp_len)
    tt = n / fs
    phase = 2 * np.pi * (f0 * tt + 0.5 * (f1 - f0) * (tt**2) / (chirp_len / fs))
    seg = amp * np.sin(phase)
    f[start:start + chirp_len] += seg

    # Normalize to unit-ish variance (helps numeric conditioning & plotting)
    std = float(np.std(f))
    if std > 0:
        f /= std
    return f


# -----------------------
# Reproducibility helpers
# -----------------------
def hash_array_sha256(x: np.ndarray) -> str:
    """Stable SHA-256 hash of a float64 array (C-contiguous bytes)."""
    arr = np.asarray(x, dtype=np.float64)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def make_manifest(
    path: str,
    *,
    N: int,
    fs: float,
    omega_list: np.ndarray,
    seed_run: int,
    seed_kernels: int,
    seed_forcing: int,
    kernels: List[KernelSpec],
    forcing_hash: str,
) -> None:
    """
    Write a JSON manifest with:
      • global sizes & sampling (N, fs), list of ω (one per kernel),
      • seeds for kernels and forcing (so both can be regenerated exactly),
      • all kernel parameters (a_m, b_m, name, meta),
      • SHA-256 of the saved forcing array (extra safety).
    """
    payload = {
        "version": 1,
        "N": int(N),
        "fs": float(fs),
        "omega_list": [float(w) for w in omega_list],
        "seed_run": int(seed_run),
        "seed_kernels": int(seed_kernels),
        "seed_forcing": int(seed_forcing),
        "forcing_sha256": forcing_hash,
        "kernels": [
            {
                "name": k.name,
                "a": np.asarray(k.a, dtype=float).tolist(),
                "b": np.asarray(k.b, dtype=float).tolist(),
                "meta": k.meta,
            }
            for k in kernels
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

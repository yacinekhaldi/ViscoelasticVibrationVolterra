#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core_experiment import (
    KernelSpec,
    build_five_kernels,
    generate_forcing,
    make_manifest,
    hash_array_sha256,
)

# -----------------------
# Hardcoded experiment IO
# -----------------------
DS_ROOT   = pathlib.Path("./datasets/synthetic").resolve()
FORCING_NPY = DS_ROOT / "forcing.npy"
FORCING_CSV = DS_ROOT / "forcing.csv"
MANIFEST    = DS_ROOT / "manifest.json"

# -----------------------
# Hardcoded experiment parameters
# -----------------------
SEED_RUN      = 202501  # run identifier (doesn't change outcomes by itself)
SEED_KERNELS  = 42      # kernel sampling seed (changing it → different kernels)
SEED_FORCING  = 99      # forcing seed (changing it → different f[n])
FS_HZ         = 500.0   # sampling rate [Hz]
N_SAMPLES     = 200_000 # choose a reasonable default (increase if you want)
OMEGA_MIN     = 4.0
OMEGA_MAX     = 9.0

def ensure_dir(p: pathlib.Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_kernel_summary_plots(kernels: list[KernelSpec], out_dir: pathlib.Path) -> None:
    """
    Save per-kernel parameters and K(t) curves (log–log) for documentation.
    These are *not* used by the solver, but helpful for reviewers & debugging.
    """
    tK = np.geomspace(1e-4, 20.0, 2000)
    for k in kernels:
        # CSV of parameters
        pd.DataFrame({"a": k.a, "b": k.b}).to_csv(out_dir / f"{k.name}_kernel_params.csv", index=False)
        # K(t) curve
        Kcurve = np.exp(-np.outer(tK, k.b)) @ k.a
        plt.figure()
        plt.loglog(tK, Kcurve)
        plt.xlabel("t (s)")
        plt.ylabel("K(t)")
        plt.title(f"Kernel {k.name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{k.name}_kernel_curve.png")
        plt.close()

def main() -> None:
    ensure_dir(DS_ROOT)

    # 1) Build kernels ONCE from SEED_KERNELS
    rng_k = np.random.default_rng(SEED_KERNELS)
    kernels = build_five_kernels(FS_HZ, rng_k)

    # 2) Build shared forcing ONCE from SEED_FORCING
    rng_f = np.random.default_rng(SEED_FORCING)
    f = generate_forcing(N_SAMPLES, FS_HZ, rng_f)
    # Save forcing vector for absolute reproducibility
    np.save(FORCING_NPY, f)
    # Optional human-readable CSV
    t = np.arange(N_SAMPLES) / FS_HZ
    pd.DataFrame({"time": t, "forcing": f}).to_csv(FORCING_CSV, index=False)

    # 3) Define ω per kernel for diversity (but fixed once recorded)
    omega_list = np.linspace(OMEGA_MIN, OMEGA_MAX, num=len(kernels))

    # 4) Manifest with *everything* needed to regenerate any experiment
    forcing_hash = hash_array_sha256(f)
    make_manifest(
        str(MANIFEST),
        N=N_SAMPLES,
        fs=FS_HZ,
        omega_list=omega_list,
        seed_run=SEED_RUN,
        seed_kernels=SEED_KERNELS,
        seed_forcing=SEED_FORCING,
        kernels=kernels,
        forcing_hash=forcing_hash,
    )

    # 5) Nice kernel summaries for documentation
    save_kernel_summary_plots(kernels, DS_ROOT)

    print(f"[OK] Wrote:\n  {FORCING_NPY}\n  {FORCING_CSV}\n  {MANIFEST}")
    print(f"[OK] Kernel summaries in: {DS_ROOT}")

if __name__ == "__main__":
    main()

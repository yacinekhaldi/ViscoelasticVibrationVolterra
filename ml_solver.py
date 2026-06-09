#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, pathlib, itertools, random, sys
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

# --------------------------
# IO LOCATIONS (hardcoded)
# --------------------------
DS_ROOT      = pathlib.Path("./datasets/synthetic").resolve()
MANIFEST     = DS_ROOT / "manifest.json"
FORCING_NPY  = DS_ROOT / "forcing.npy"

RESULTS_ROOT = pathlib.Path("./results/kernels").resolve()
ML_ROOT      = pathlib.Path("./results/ml_v8").resolve()

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

# --- tiny Tee to log console to file (kept) ---
class _Tee:
    def __init__(self, *files):
        self._files = files
    def write(self, data):
        for f in self._files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self._files:
            f.flush()

# --------------------------
# EXPERIMENT CONFIG (edit)
# --------------------------
SEED               = 1337
METHOD_FOR_DATA    = "semi_implicit__prony_state"  # which numerical method’s CSV to consume
FS_FALLBACK        = 500.0   # will be overridden by manifest fs
DT_FALLBACK        = 1.0 / FS_FALLBACK

# Windowing
WINDOW             = 512      # input lookback sequence length
HORIZON            = 64       # forecast steps ahead (multi-step)
STRIDE             = 32       # stride between windows (for more samples)

# Model sizes
HIDDEN             = 128
PRONY_M            = 24       # number of exponential modes in NPSSL
TCN_CHANNELS       = 64       # small causal CNN over features

# Optimization
EPOCHS             = 30
BATCH_SIZE         = 64
LR                 = 1e-3
WEIGHT_DECAY       = 1e-5

# Loss weights (base targets)
L_FORECAST         = 1.0      # displacement forecast loss
L_CLASS            = 0.5      # kernel family classification
L_A_REG            = 0.5      # amplitude regression (Prony amplitudes)
L_PHYS_TARGET      = 0.9      # (v2) stronger target physics residual weight

# (v2) Simple curriculum on physics residual:
PHYS_WARMUP_EPOCHS = 8        # small residual weight at first, then ramp
L_PHYS_WARMUP      = 0.1      # initial residual weight during warm-up

# Train/Val/Test split fraction (by windows, stratified by kernel family)
TRAIN_FRAC         = 0.7
VAL_FRAC           = 0.15
TEST_FRAC          = 0.15

# --------------------------
# UTILITIES
# --------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_manifest() -> Dict[str, Any]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest missing at {MANIFEST}. Run the generator first.")
    with open(MANIFEST, "r", encoding="utf-8") as f:
        return json.load(f)

def list_kernels(mani: Dict[str, Any]) -> List[str]:
    return [k["name"] for k in mani["kernels"]]

def load_solver_csv(kernel_name: str, method: str) -> pd.DataFrame:
    csv = RESULTS_ROOT / kernel_name / method / "timeseries.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Solver CSV not found: {csv}")
    return pd.read_csv(csv)

def plot_forcing_spectrum(f: np.ndarray, fs: float, out_png: pathlib.Path):
    N = f.shape[0]
    F = np.fft.rfft(f)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    mag = np.abs(F)
    plt.figure()
    plt.plot(freqs, mag)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|F(f)|")
    plt.title("Forcing Spectrum")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# --------------------------
# DATASET CONSTRUCTION
# --------------------------
class WindowedViscoDataset(Dataset):
    """
    Builds (X,y) windows from solver outputs for multi-task learning:
      Inputs X[n] include: [forcing, u, v, a, H] over past WINDOW steps.
      Tasks:
        1) Forecast: predict u over next HORIZON steps (sequence-to-sequence).
        2) Classification: kernel family id (0..4).
        3) Regression: Prony amplitudes a_m (true from manifest for this kernel).
        4) Physics residual: computed on-the-fly inside the model via NPSSL.
    We collect windows across all kernel families to enable generalization.
    """
    def __init__(self, mani: Dict[str, Any], method: str, window: int, horizon: int, stride: int):
        super().__init__()
        self.mani = mani
        self.method = method
        self.window = window
        self.horizon = horizon
        self.stride = stride

        self.fs = float(mani.get("fs", FS_FALLBACK))
        self.dt = 1.0 / self.fs

        # map kernel names to class ids
        self.kernels_json = mani["kernels"]
        self.kernel_names = [k["name"] for k in self.kernels_json]
        self.cls_map = {name: i for i, name in enumerate(self.kernel_names)}

        # true amplitudes (for regression head)
        self.true_a_by_kernel = {k["name"]: np.array(k["a"], dtype=np.float32) for k in self.kernels_json}
        self.true_b_by_kernel = {k["name"]: np.array(k["b"], dtype=np.float32) for k in self.kernels_json}
        self.max_M = max(len(k["a"]) for k in self.kernels_json)  # for padding to uniform length

        # load dataframes per kernel
        self.frames = {}
        for name in self.kernel_names:
            self.frames[name] = load_solver_csv(name, self.method)

        # materialize window indices (kernel, start_idx)
        self.index: List[Tuple[str, int]] = []
        for name in self.kernel_names:
            df = self.frames[name]
            N = len(df)
            last_start = N - (self.window + self.horizon)
            if last_start <= 0: 
                continue
            for s in range(0, last_start, self.stride):
                self.index.append((name, s))

        # stratify roughly by kernel family
        # (we’ll split later via random_split on indices)
        self.n_samples = len(self.index)

    def __len__(self):
        return self.n_samples

    def _pad_to_M(self, a_vec: np.ndarray) -> np.ndarray:
        """Pad per-kernel true amplitudes to common length max_M for consistent regression targets."""
        out = np.zeros(self.max_M, dtype=np.float32)
        out[: len(a_vec)] = a_vec
        return out

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        kname, s = self.index[idx]
        df = self.frames[kname]  # columns: time, forcing, disp, vel, acc, hist

        w = self.window
        h = self.horizon

        seg_in = df.iloc[s : s+w]
        seg_out = df.iloc[s+w : s+w+h]

        X = np.stack([
            seg_in["forcing"].to_numpy(dtype=np.float32),
            seg_in["disp"].to_numpy(dtype=np.float32),
            seg_in["vel"].to_numpy(dtype=np.float32),
            seg_in["acc"].to_numpy(dtype=np.float32),
            seg_in["hist"].to_numpy(dtype=np.float32),
        ], axis=-1)  # [w, 5]

        y_fore = seg_out["disp"].to_numpy(dtype=np.float32)  # [h]
        y_cls  = self.cls_map[kname]
        y_a    = self._pad_to_M(self.true_a_by_kernel[kname])  # [max_M]

        M_true = len(self.true_a_by_kernel[kname])
        mask_a = np.zeros(self.max_M, dtype=np.float32)
        mask_a[:M_true] = 1.0
        return {
            "X": X,
            "y_fore": y_fore,
            "y_cls": y_cls,
            "a_true": y_a,
            "mask_a": mask_a, 
            "kname": kname,
        }

def train_val_test_split(ds: WindowedViscoDataset):
    """Simple split by shuffled indices, preserving class balance approximately due to many windows."""
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)

    n_tr = int(TRAIN_FRAC * n)
    n_va = int(VAL_FRAC * n)
    tr_idx = idx[:n_tr]
    va_idx = idx[n_tr : n_tr + n_va]
    te_idx = idx[n_tr + n_va :]

    class _Subset(Dataset):
        def __init__(self, base: Dataset, indices: np.ndarray):
            self.base = base
            self.indices = indices
        def __len__(self): return len(self.indices)
        def __getitem__(self, i): return self.base[int(self.indices[i])]

    return _Subset(ds, tr_idx), _Subset(ds, va_idx), _Subset(ds, te_idx)

# --------------------------
# MODEL: Causal TCN + NPSSL
# --------------------------
class CausalTCN(nn.Module):
    """Small causal 1D CNN over time. Input shape: [B, W, F]; output: [B, W, C]."""
    def __init__(self, in_ch=5, channels=64):
        super().__init__()
        pad1, dil1 = 2, 2
        pad2, dil2 = 4, 4
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, channels, kernel_size=3, padding=1),  # causal-ish first conv
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad1, dilation=dil1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad2, dilation=dil2),
            nn.GELU(),
        )
    def forward(self, x):  # x: [B,W,F]
        x = x.transpose(1, 2)     # -> [B,F,W]
        y = self.net(x)           # -> [B,C,W]
        return y.transpose(1, 2)  # -> [B,W,C]

class NPSSL(nn.Module):
    """
    Neural Prony State-Space Layer
    - Fixed log-spaced decay rates b[m] covering [tmin, tmax] ~ window scale.
    - Learnable nonnegative amplitudes a[m] via softplus.
    - Causal state update produces H[n] from velocity v[n].
    """
    def __init__(self, prony_M: int, dt: float, tmin=None, tmax=None):
        super().__init__()
        self.M = prony_M
        self.dt = dt
        # set a broad log-spaced grid of b's; cover ~ [dt*5, 10s] by default
        tmin = 5*dt if tmin is None else tmin
        tmax = 10.0 if tmax is None else tmax
        b = np.geomspace(1.0/tmax, 1.0/max(tmin, 1e-4), num=prony_M).astype(np.float32)
        self.register_buffer("b_grid", torch.from_numpy(b))  # [M]
        self.a_raw = nn.Parameter(torch.full((prony_M,), -2.0))  # start small (softplus ~0)
        self.softplus = nn.Softplus()

    def forward(self, v_seq: torch.Tensor):
        """
        v_seq: [B, W] velocity sequence (past window)
        returns: H_seq [B, W], a_pred [B,M] (same a across time in this simple version)
        """
        B, W = v_seq.shape
        a = self.softplus(self.a_raw)                      # [M] nonnegative
        decay = torch.exp(-self.b_grid * self.dt)          # [M]
        z = torch.zeros((B, self.M), device=v_seq.device)  # states

        H = []
        for n in range(W):
            z = z * decay + a * (self.dt * v_seq[:, n:n+1])  # broadcasting a over batch
            H.append(z.sum(dim=1, keepdim=True))
        H_seq = torch.cat(H, dim=1)  # [B, W]
        a_pred = a.unsqueeze(0).repeat(B, 1)  # [B,M]
        return H_seq, a_pred

    # For analysis/plots
    def kernel_curve(self, t_grid: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            a = self.softplus(self.a_raw).detach().cpu().numpy()  # [M]
            b = self.b_grid.detach().cpu().numpy()
            K = np.exp(-np.outer(t_grid, b)) @ a
            return K

class GRUForecaster(nn.Module):
    """Pure-ML baseline forecaster (no NPSSL). Uses [f,u,v,a,H] to predict next H steps of u."""
    def __init__(self, in_feats: int, hidden: int, horizon: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_feats, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon))

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X: [B,W,F]
        y, _ = self.gru(X)                 # [B,W,H]
        h = y[:, -1, :]                    # last
        return self.head(h)                # [B,H]

class TinyTransformerForecaster(nn.Module):
    """Pure-ML baseline: 2-layer Transformer encoder on time axis; predicts next H steps of u."""
    def __init__(self, in_feats: int, d_model: int, nhead: int, num_layers: int, horizon: int, dim_ff: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_feats, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon))

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X: [B,W,F]
        Z = self.proj(X)          # [B,W,D]
        E = self.enc(Z)           # [B,W,D]
        h = E[:, -1, :]           # last
        return self.head(h)       # [B,H]


# (v2) Residual utilities: smoothing & second derivative
def _moving_avg_kernel(win: int) -> torch.Tensor:
    k = torch.ones(win, dtype=torch.float32) / float(win)
    return k.view(1, 1, -1)

def _smooth_1d(x: torch.Tensor, win: int = 9) -> torch.Tensor:
    """Simple causal-ish smoothing via 1D conv with small moving average kernel (no future leak by padding left)."""
    if win <= 1:
        return x
    pad = (win - 1, 0)  # pad left only to reduce future peeking
    k = _moving_avg_kernel(win).to(x.device)  # [1,1,win]
    x_ = F.pad(x.unsqueeze(1), pad=(pad[0], pad[1]))  # [B,1,W+pad]
    y = F.conv1d(x_, k).squeeze(1)
    return y

def _get_mask_a(batch, ya, device, M):
    """Return amplitude mask with correct device/dtype/shape."""
    t = batch.get("mask_a", None)
    if t is None:
        return torch.ones_like(ya, device=device)[:, :M]
    t = t.to(device)
    if t.dtype != torch.float32:
        t = t.float()
    return t[:, :M]

def _second_derivative(x: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Savitzky–Golay-like 2nd derivative (symmetric 7-tap stencil).
    Output length = W-6 (crop 3 on both sides) to avoid edge artifacts.
    """
    # 7-point second-derivative coefficients (approx. SG poly fit)
    # [-1, 12, -39, 56, -39, 12, -1] / (6 * dt^2)
    coeff = torch.tensor([-1, 12, -39, 56, -39, 12, -1], dtype=x.dtype, device=x.device) / (6.0 * dt * dt)
    coeff = coeff.view(1, 1, -1)  # [1,1,7]
    # depthwise conv over time
    xx = x.unsqueeze(1)           # [B,1,W]
    u2_full = torch.nn.functional.conv1d(xx, coeff, padding=0)  # [B,1,W-6]
    return u2_full.squeeze(1)     # [B, W-6]

def _parabolic_interp(mag_row, idx):
    # idx ∈ [1, F-2]; returns fractional offset δ ∈ [-0.5, 0.5]
    a = mag_row[idx-1]; b = mag_row[idx]; c = mag_row[idx+1]
    denom = (a - 2*b + c)
    if denom.abs() < 1e-12:
        return 0.0
    return 0.5 * (a - c) / denom

def _estimate_omega_batch(u: torch.Tensor, dt: float, k: int = 1) -> torch.Tensor:
    """
    Estimate per-sample natural frequency from a displacement window u [B, W]
    using the dominant peak of |FFT(u)| (ignore DC). Returns ω [B, 1].
    k: take the k-th strongest non-DC peak to avoid trivial DC picks.
    """
    # rFFT: [B, F], freqs: [F]
    U = torch.fft.rfft(u, dim=1)                       # complex
    mag = torch.abs(U)                                 # [B, F]
    # zero out DC
    mag[:, 0] = 0.0
    # take peak index per row
    # (optionally the k-th strongest for robustness)
    vals, idxs = torch.topk(mag, k=max(1, k), dim=1)
    peak_idx = idxs[:, min(k-1, idxs.size(1)-1)]
    # clamp to allow ±1 neighborhood
    peak_idx = torch.clamp(peak_idx, 1, mag.size(1)-2)
    # parabolic offset per sample
    offsets = []
    for i in range(mag.size(0)):
        offsets.append(_parabolic_interp(mag[i], int(peak_idx[i])) )
    delta = torch.tensor(offsets, device=mag.device, dtype=mag.dtype)
    # refined frequency
    bin_width = (1.0/dt) / (2*(mag.size(1)-1))     # fs/2 divided by (#rfft bins-1)
    f_refined = peak_idx * bin_width + delta * bin_width
    omega = 2.0 * torch.pi * f_refined

    return omega.view(-1, 1)                                   # [B,1]


class MultiTaskModel(nn.Module):
    """
    End-to-end:
      Inputs: X[:, :, 0..4] = [forcing, u, v, a, H] over a past window (length W)
      Encoder: small Causal TCN
      Physics: NPSSL over v -> \hat H
      Heads:   forecast u[t+1..t+H], classify kernel family, regress amplitudes
      Residual: r = u'' + w^2 u + \hat H - f  (computed over the window, smoothed u'')
    """
    def __init__(self, n_features: int, horizon: int, n_classes: int, prony_M: int, dt: float, hidden: int = HIDDEN, tcn_channels: int = TCN_CHANNELS):
        super().__init__()
        self.horizon = horizon
        self.dt = dt
        self.enc = CausalTCN(in_ch=n_features, channels=tcn_channels)
        self.npssl = NPSSL(prony_M=prony_M, dt=dt)

        # simple heads
        self.head_fore = nn.Sequential(
            nn.Linear(tcn_channels, hidden), nn.GELU(),
            nn.Linear(hidden, horizon)
        )
        self.head_cls = nn.Sequential(
            nn.Linear(tcn_channels, hidden), nn.GELU(),
            nn.Linear(hidden, n_classes)
        )
        self.head_amp = nn.Sequential(
            nn.Linear(tcn_channels, hidden), nn.GELU(),
            nn.Linear(hidden, prony_M), nn.Softplus()  # non-negative amplitudes
        )

    def forward(self, X: torch.Tensor, dt: float, omega) -> dict:
        """
        X: [B, W, 5] with channels: [forcing=f, u, v, a, ...]
        omega: scalar (float or 0-dim tensor) or per-sample tensor [B,1]
        Returns:
        - "y_fore": [B, H]
        - "y_cls":  [B, n_classes]
        - "a_pred": [B, M]
        - "residual": [B, W-6]  (physics residual using 7-tap 2nd derivative)
        """
        B, W, C = X.shape
        device = X.device

        # Unpack channels
        forcing = X[:, :, 0]   # f[n]
        u       = X[:, :, 1]   # displacement
        v       = X[:, :, 2]   # velocity
        a_meas  = X[:, :, 3]   # acceleration (measured), optional downstream use

        # --- Encoder (Causal TCN) ---
        # self.enc expects [B,W,F] and returns [B,W,C]
        feats  = self.enc(X)          # [B, W, tcn_channels]
        pooled = feats[:, -1, :]      # last timestep (causal)

        # --- Heads ---
        y_fore = self.head_fore(pooled)   # [B, HORIZON]
        y_cls  = self.head_cls(pooled)    # [B, n_classes]
        a_pred = self.head_amp(pooled)    # [B, prony_M] (Softplus)

        # --- NPSSL: H[v] over the full window ---
        H_hat, _aux = self.npssl(v)       # [B, W]

        # --- Physics residual with 7-tap second derivative (length W-6) ---
        u_s    = _smooth_1d(u, win=9)
        u2     = _second_derivative(u_s, dt=dt)
        H_crop = H_hat[:, 3:-3]                # [B, W-6]
        u_crop = u[:, 3:-3]                    # [B, W-6]
        f_crop = forcing[:, 3:-3]              # [B, W-6]

        # Broadcast omega to [B,1]
        if isinstance(omega, (float, int)):
            omega_b = torch.full((B, 1), float(omega), device=device, dtype=u.dtype)
        else:
            omega_t = torch.as_tensor(omega, device=device, dtype=u.dtype)
            if omega_t.dim() == 0:
                omega_b = omega_t.view(1, 1).expand(B, 1)
            elif omega_t.dim() == 1:
                assert omega_t.shape[0] == B, "omega shape [B] mismatch with batch size"
                omega_b = omega_t.view(B, 1)
            elif omega_t.dim() == 2:
                assert omega_t.shape[0] == B and omega_t.shape[1] == 1, "omega must be [B,1]"
                omega_b = omega_t
            else:
                raise ValueError("omega must be scalar or shape [B] / [B,1]")

        # Residual: u'' + ω^2 u + H[v] - f  (all cropped to W-6)
        residual = u2 + (omega_b**2) * u_crop + H_crop - f_crop   # [B, W-6]

        return {
            "y_fore": y_fore,
            "y_cls":  y_cls,
            "a_pred": a_pred,
            "residual": residual,
            # optional: "_H_hat": H_hat, "_u2": u2, "_omega": omega_b
        }


# --------------------------
# MAIN TRAIN/TEST
# --------------------------
def run_experiment():
    """
    v8: Proposal-compliant experiment runner
    ---------------------------------------
    Trains/evaluates NPSSL model (your core model) AND pure-ML baselines (GRU, Transformer)
    for each numerical method already exported by numerical_solver.py:
        ["semi_implicit__prony_state", "semi_implicit__fft_conv",
         "verlet__prony_state", "verlet__fft_conv"]
    Keeps all your current exports and adds comparative artifacts in analysis/.
    """
    import math

    # ----------------- setup & dirs -----------------
    set_all_seeds(SEED)
    global ML_ROOT
    ML_ROOT = pathlib.Path("./results/ml_v8").resolve()   # v8 output root
    ensure_dir(ML_ROOT); ensure_dir(ML_ROOT / "checkpoints")
    ensure_dir(ML_ROOT / "arrays"); ensure_dir(ML_ROOT / "samples"); ensure_dir(ML_ROOT / "analysis")
    ensure_dir(ML_ROOT / "baselines")

    # tee logs (keep)
    _log_f = open(ML_ROOT / "output.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_f)
    sys.stderr = _Tee(sys.stderr, _log_f)

    # ----------------- global manifest / plots -----------------
    mani = load_manifest()
    fs = float(mani.get("fs", FS_FALLBACK)); dt = 1.0 / fs
    if FORCING_NPY.exists():
        f = np.load(FORCING_NPY); plot_forcing_spectrum(f, fs, ML_ROOT / "spectrum_forcing.png")

    # Methods already exported by numerical_solver.py (no re-solving)  :contentReference[oaicite:2]{index=2}
    METHODS = [
        "semi_implicit__prony_state",
        "semi_implicit__fft_conv",
        "verlet__prony_state",
        "verlet__fft_conv",
    ]

    # ---- helper: build DataLoaders for a method ----
    def _make_loaders(method: str):
        ds = WindowedViscoDataset(mani, method, WINDOW, HORIZON, STRIDE)
        tr, va, te = train_val_test_split(ds)

        # class-balanced train sampler (unchanged)
        labels_tr = np.array([tr[i]["y_cls"] for i in range(len(tr))], dtype=int)
        class_counts = np.bincount(labels_tr, minlength=len(ds.kernel_names))
        class_weights_np = (1.0 / np.maximum(class_counts, 1)).astype(np.float64)
        class_weights_np *= (len(class_weights_np) / class_weights_np.sum())
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(class_weights_np[labels_tr], dtype=torch.double),
            num_samples=len(labels_tr),
            replacement=True
        )
        dl_tr = DataLoader(tr, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
        dl_va = DataLoader(va, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        dl_te = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        return ds, dl_tr, dl_va, dl_te, class_weights_np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- storage for per-method summary rows (NPSSL + baselines) ----
    rows_all_methods = []

    # =================================================================
    # loop over numerical methods (proposal comparison without re-solving)
    # =================================================================
    for METHOD_FOR_DATA in METHODS:
        print(f"\n========== METHOD: {METHOD_FOR_DATA} ==========")
        ds, dl_tr, dl_va, dl_te, class_weights_np = _make_loaders(METHOD_FOR_DATA)
        PRONY_M_local = ds.max_M
        n_classes = len(ds.kernel_names)

        # ----------------- kernel-aware weights (as in v7) -----------------
        KPHYS = {"maxwell_single":2.0,"prony_biexp":1.5,"prony_triexp":1.0,"stretched_exp_approx":1.0,"powerlaw_approx":1.0}
        KFORE = {"maxwell_single":1.5,"prony_biexp":1.1,"prony_triexp":1.0,"stretched_exp_approx":1.2,"powerlaw_approx":1.0}
        KAREG = {"maxwell_single":1.0,"prony_biexp":1.0,"prony_triexp":1.0,"stretched_exp_approx":1.3,"powerlaw_approx":1.3}
        def _kernel_w(klist, table): return torch.tensor([table.get(str(n),1.0) for n in klist], device=device, dtype=torch.float32).view(-1,1)
        def _kernel_phys(klist):     return _kernel_w(klist, KPHYS)

        # ----------------- NPSSL model (+ three heads) -----------------
        model = MultiTaskModel(
            n_features=5, horizon=HORIZON, n_classes=n_classes,
            prony_M=PRONY_M_local, dt=dt, hidden=HIDDEN, tcn_channels=TCN_CHANNELS
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        cw = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
        cw = cw / cw.mean()

        CLS_WARMUP_EPOCHS, AREG_WARMUP_EPOCHS = 5, 3
        log_rows = []; best_va_core = float("inf")
        ck_dir = ML_ROOT / "checkpoints" / METHOD_FOR_DATA; ensure_dir(ck_dir)
        best_path = ck_dir / "best_model.pt"

        # ----------------- TRAIN (NPSSL) -----------------
        for epoch in range(1, EPOCHS + 1):
            # ramp physics weight
            if epoch <= PHYS_WARMUP_EPOCHS: L_PHYS = L_PHYS_WARMUP
            else:
                alpha = min(1.0, (epoch - PHYS_WARMUP_EPOCHS) / max(1, EPOCHS - PHYS_WARMUP_EPOCHS))
                L_PHYS = L_PHYS_WARMUP + alpha * (L_PHYS_TARGET - L_PHYS_WARMUP)
            w_cls = 0.0 if epoch < CLS_WARMUP_EPOCHS else L_CLASS
            w_a   = 0.0 if epoch < AREG_WARMUP_EPOCHS else L_A_REG

            # ---- train epoch ----
            model.train(); tr_loss=tr_fore=tr_cls=tr_areg=tr_phys=0.0; seen=0
            for batch in dl_tr:
                X,yf,yc = batch["X"].to(device), batch["y_fore"].to(device), batch["y_cls"].to(device)
                ya = batch["a_true"].to(device)[:, :PRONY_M_local]
                ma = _get_mask_a(batch, ya, device, PRONY_M_local)
                klist = batch["kname"]
                omega_b = _estimate_omega_batch(X[:, :, 1], dt, k=2)

                out = model(X, dt, omega_b)
                loss_cls = torch.tensor(0.0, device=device); loss_a = torch.tensor(0.0, device=device)

                per_fore = ((out["y_fore"]-yf)**2).mean(dim=1, keepdim=True); loss_fore = (_kernel_w(klist,KFORE)*per_fore).mean()
                if w_cls>0: loss_cls = F.cross_entropy(out["y_cls"], yc, weight=cw)
                if w_a>0:
                    per_areg = (torch.abs(out["a_pred"]-ya)*ma).sum(dim=1, keepdim=True)/(ma.sum(dim=1,keepdim=True)+1e-8)
                    loss_a = (_kernel_w(klist,KAREG)*per_areg).mean()

                forcing,u,v = X[:,:,0],X[:,:,1],X[:,:,2]
                u2 = _second_derivative(_smooth_1d(u,win=9), dt=dt)
                H_hat,_ = model.npssl(v); Hc=H_hat[:,3:-3]; uc=u[:,3:-3]; fc=forcing[:,3:-3]
                denom = (torch.abs(u2)+torch.abs((omega_b**2)*uc)+torch.abs(Hc)+torch.abs(fc)).mean(dim=1,keepdim=True)+1e-8
                r_norm = out["residual"]/denom
                loss_phys = (_kernel_phys(klist)*F.smooth_l1_loss(r_norm, torch.zeros_like(r_norm), reduction="none").mean(dim=1,keepdim=True)).mean()

                loss = (L_FORECAST*loss_fore)+(w_cls*loss_cls)+(w_a*loss_a)+(L_PHYS*loss_phys)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

                bsz=X.size(0); tr_loss+=float(loss)*bsz; tr_fore+=float(loss_fore)*bsz; tr_cls+=float(loss_cls)*bsz; tr_areg+=float(loss_a)*bsz; tr_phys+=float(loss_phys)*bsz; seen+=bsz

            tr_loss/=seen; tr_fore/=seen; tr_cls/=seen; tr_areg/=seen; tr_phys/=seen

            # ---- validation ----
            model.eval(); va_loss=va_fore=va_cls=va_areg=va_phys=0.0; seen=0
            with torch.no_grad():
                for batch in dl_va:
                    X,yf,yc = batch["X"].to(device), batch["y_fore"].to(device), batch["y_cls"].to(device)
                    ya = batch["a_true"].to(device)[:, :PRONY_M_local]
                    ma = _get_mask_a(batch, ya, device, PRONY_M_local)
                    klist = batch["kname"]; omega_b = _estimate_omega_batch(X[:, :, 1], dt, k=2)
                    out = model(X, dt, omega_b)
                    loss_cls = torch.tensor(0.0, device=device); loss_a = torch.tensor(0.0, device=device)

                    per_fore = ((out["y_fore"]-yf)**2).mean(dim=1, keepdim=True); loss_fore = (_kernel_w(klist,KFORE)*per_fore).mean()
                    if w_cls>0: loss_cls = F.cross_entropy(out["y_cls"], yc, weight=cw)
                    if w_a>0:
                        per_areg = (torch.abs(out["a_pred"]-ya)*ma).sum(dim=1, keepdim=True)/(ma.sum(dim=1,keepdim=True)+1e-8)
                        loss_a = (_kernel_w(klist,KAREG)*per_areg).mean()

                    forcing,u,v = X[:,:,0],X[:,:,1],X[:,:,2]
                    u2 = _second_derivative(_smooth_1d(u,win=9), dt=dt)
                    H_hat,_ = model.npssl(v); Hc=H_hat[:,3:-3]; uc=u[:,3:-3]; fc=forcing[:,3:-3]
                    denom=(torch.abs(u2)+torch.abs((omega_b**2)*uc)+torch.abs(Hc)+torch.abs(fc)).mean(dim=1,keepdim=True)+1e-8
                    r_norm = out["residual"]/denom
                    loss_phys=(_kernel_phys(klist)*F.smooth_l1_loss(r_norm, torch.zeros_like(r_norm), reduction="none").mean(dim=1,keepdim=True)).mean()

                    loss=(L_FORECAST*loss_fore)+(w_cls*loss_cls)+(w_a*loss_a)+(L_PHYS*loss_phys)
                    bsz=X.size(0); va_loss+=float(loss)*bsz; va_fore+=float(loss_fore)*bsz; va_cls+=float(loss_cls)*bsz
                    va_areg+=float(loss_a)*bsz; va_phys+=float(loss_phys)*bsz; seen+=bsz

            va_loss/=seen; va_fore/=seen; va_cls/=seen; va_areg/=seen; va_phys/=seen
            va_core=(L_FORECAST*va_fore)+(max(w_cls,0)*va_cls)+(max(w_a,0)*va_areg)
            sched.step(va_core)
            if va_core<best_va_core:
                best_va_core=va_core
                torch.save({"model":model.state_dict(),
                            "config":{"WINDOW":WINDOW,"HORIZON":HORIZON,"PRONY_M":PRONY_M_local,"n_classes":n_classes,"dt":dt}},
                           best_path)
            log_rows.append({"epoch":epoch,"train_loss":tr_loss,"val_loss":va_loss,"val_core":va_core,
                             "tr_fore":tr_fore,"tr_cls":tr_cls,"tr_areg":tr_areg,"tr_phys":tr_phys,
                             "va_fore":va_fore,"va_cls":va_cls,"va_areg":va_areg,"va_phys":va_phys,
                             "L_PHYS":L_PHYS,"w_cls":w_cls,"w_a":w_a})
            print(f"[{METHOD_FOR_DATA}] epoch {epoch:02d} train={tr_loss:.4f} val={va_loss:.4f} core={va_core:.4f}")

        # save curves per method
        df_log = pd.DataFrame(log_rows)
        ensure_dir(ML_ROOT / "logs"); df_log.to_csv(ML_ROOT / "logs" / f"training_log__{METHOD_FOR_DATA}.csv", index=False)
        plt.figure(); plt.plot(df_log["epoch"], df_log["train_loss"], label="train")
        plt.plot(df_log["epoch"], df_log["val_loss"], label="val")
        plt.plot(df_log["epoch"], df_log["val_core"], label="val_core (no physics)", linestyle="--")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
        plt.savefig(ML_ROOT / f"loss_curves__{METHOD_FOR_DATA}.png"); plt.close()

        # ----------------- TEST (NPSSL) -----------------
        # (Identical to your v7 test section; writes the same arrays/plots
        #  but now with method suffix to avoid overwriting.)
        def _suffix(name: str) -> pathlib.Path: return (ML_ROOT / name).with_name(f"{(ML_ROOT / name).stem}__{METHOD_FOR_DATA}{(ML_ROOT / name).suffix}")

        if best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device)["model"])
        model.eval()

        # collectors
        y_true_all=[]; y_pred_all=[]; cls_true_all=[]; cls_pred_all=[]; a_true_all=[]; a_pred_all=[]
        residuals_all=[]; residuals_norm_all=[]; residuals_acc_all=[]
        residuals_by_kernel={k:[] for k in ds.kernel_names}; residuals_norm_by_kernel={k:[] for k in ds.kernel_names}; residuals_acc_by_kernel={k:[] for k in ds.kernel_names}
        reg_err_by_kernel={k:[] for k in ds.kernel_names}
        forecast_mse_by_kernel={k:[] for k in ds.kernel_names}; areg_mse_by_kernel={k:[] for k in ds.kernel_names}
        cls_tp_by_kernel={k:0 for k in ds.kernel_names}; cls_support_by_kernel={k:0 for k in ds.kernel_names}

        with torch.no_grad():
            for batch in dl_te:
                X,yf,yc = batch["X"].to(device), batch["y_fore"].to(device), batch["y_cls"].to(device)
                ya = batch["a_true"].to(device)[:, :PRONY_M_local]
                ma = _get_mask_a(batch, ya, device, PRONY_M_local)
                klist = batch["kname"]; omega_b = _estimate_omega_batch(X[:, :, 1], dt, k=2)
                out = model(X, dt, omega_b)

                y_true_all.append(yf.cpu().numpy()); y_pred_all.append(out["y_fore"].cpu().numpy())
                pred_cls = out["y_cls"].argmax(dim=1).cpu().numpy()
                cls_true_all.append(yc.cpu().numpy()); cls_pred_all.append(pred_cls)
                a_true_all.append(ya.cpu().numpy()); a_pred_all.append(out["a_pred"].cpu().numpy())

                abs_err = (torch.abs(out["a_pred"]-ya)*ma).cpu().numpy()
                for i,kname in enumerate(klist): reg_err_by_kernel[kname].append(abs_err[i])

                yc_np = yc.cpu().numpy()
                for i,kname in enumerate(klist):
                    cls_support_by_kernel[kname]+=1
                    if int(pred_cls[i])==int(yc_np[i]): cls_tp_by_kernel[kname]+=1

                per_fore = ((out["y_fore"]-yf)**2).mean(dim=1).cpu().numpy()
                for i,kname in enumerate(klist): forecast_mse_by_kernel[kname].append(float(per_fore[i]))
                sq = (((out["a_pred"]-ya)*ma)**2); per_areg = (sq.sum(dim=1)/(ma.sum(dim=1)+1e-8)).cpu().numpy()
                for i,kname in enumerate(klist): areg_mse_by_kernel[kname].append(float(per_areg[i]))

                res_np = out["residual"].cpu().numpy(); residuals_all.append(res_np)
                for i,kname in enumerate(klist): residuals_by_kernel[kname].append(res_np[i])

                forcing,u,v = X[:,:,0],X[:,:,1],X[:,:,2]
                u2 = _second_derivative(_smooth_1d(u,win=9), dt=dt)
                H_hat,_ = model.npssl(v); Hc=H_hat[:,3:-3]; uc=u[:,3:-3]; fc=forcing[:,3:-3]
                denom=(torch.abs(u2)+torch.abs((omega_b**2)*uc)+torch.abs(Hc)+torch.abs(fc)).mean(dim=1,keepdim=True)+1e-8
                r_norm = (out["residual"]/denom).cpu().numpy(); residuals_norm_all.append(r_norm)
                for i,kname in enumerate(klist): residuals_norm_by_kernel[kname].append(r_norm[i])

                a_meas = X[:,:,3]; ac=a_meas[:,3:-3]; r_acc=(ac+(omega_b**2)*uc+Hc-fc).cpu().numpy()
                residuals_acc_all.append(r_acc)
                for i,kname in enumerate(klist): residuals_acc_by_kernel[kname].append(r_acc[i])

        # concat
        y_true_all=np.concatenate(y_true_all,axis=0); y_pred_all=np.concatenate(y_pred_all,axis=0)
        cls_true_all=np.concatenate(cls_true_all,axis=0); cls_pred_all=np.concatenate(cls_pred_all,axis=0)
        a_true_all=np.concatenate(a_true_all,axis=0); a_pred_all=np.concatenate(a_pred_all,axis=0)
        residuals_all=np.concatenate(residuals_all,axis=0); residuals_norm_all=np.concatenate(residuals_norm_all,axis=0); residuals_acc_all=np.concatenate(residuals_acc_all,axis=0)

        # save arrays (method-suffixed; your original names remain but per-method)
        arr_dir = ML_ROOT / "arrays"; ensure_dir(arr_dir)
        np.save(arr_dir / f"y_true_forecast__{METHOD_FOR_DATA}.npy", y_true_all)
        np.save(arr_dir / f"y_pred_forecast__{METHOD_FOR_DATA}.npy", y_pred_all)
        np.save(arr_dir / f"cls_true__{METHOD_FOR_DATA}.npy", cls_true_all)
        np.save(arr_dir / f"cls_pred__{METHOD_FOR_DATA}.npy", cls_pred_all)
        np.save(arr_dir / f"a_true__{METHOD_FOR_DATA}.npy", a_true_all)
        np.save(arr_dir / f"a_pred__{METHOD_FOR_DATA}.npy", a_pred_all)
        np.save(arr_dir / f"residuals__{METHOD_FOR_DATA}.npy", residuals_all)
        np.save(arr_dir / f"residuals_normalized__{METHOD_FOR_DATA}.npy", residuals_norm_all)
        np.save(arr_dir / f"residuals_acc__{METHOD_FOR_DATA}.npy", residuals_acc_all)

        # confusion matrix/plot
        acc = accuracy_score(cls_true_all, cls_pred_all)
        cm = confusion_matrix(cls_true_all, cls_pred_all, labels=list(range(len(ds.kernel_names))))
        disp = ConfusionMatrixDisplay(cm, display_labels=ds.kernel_names)
        plt.figure(figsize=(6,6)); disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Kernel Family Confusion Matrix — {METHOD_FOR_DATA}")
        plt.tight_layout(); plt.savefig(ML_ROOT / f"confusion_matrix__{METHOD_FOR_DATA}.png"); plt.close()
        pd.DataFrame(cm, index=ds.kernel_names, columns=ds.kernel_names).to_csv(ML_ROOT / f"confusion_matrix__{METHOD_FOR_DATA}.csv")

        # forecast/amp/residual plots (method-suffixed)
        plt.figure(); sl=min(200, y_true_all.shape[0]*y_true_all.shape[1])
        plt.plot(y_true_all.ravel()[:sl], label="True", alpha=0.7); plt.plot(y_pred_all.ravel()[:sl], label="Pred", alpha=0.7)
        plt.legend(); plt.title(f"Forecast vs. Ground truth (slice) — {METHOD_FOR_DATA}")
        plt.tight_layout(); plt.savefig(ML_ROOT / f"forecast_vs_true__{METHOD_FOR_DATA}.png"); plt.close()

        plt.figure(); pts=min(5000, a_true_all.size)
        plt.scatter(a_true_all.ravel()[:pts], a_pred_all.ravel()[:pts], alpha=0.4, s=8)
        plt.xlabel("True amplitudes"); plt.ylabel("Predicted amplitudes"); plt.title(f"Amplitude regression (slice) — {METHOD_FOR_DATA}")
        plt.tight_layout(); plt.savefig(ML_ROOT / f"amplitude_regression__{METHOD_FOR_DATA}.png"); plt.close()

        plt.figure(); plt.hist(residuals_all.ravel(), bins=50, alpha=0.8)
        plt.title(f"Physics residual distribution (raw) — {METHOD_FOR_DATA}")
        plt.tight_layout(); plt.savefig(ML_ROOT / f"residual_hist__{METHOD_FOR_DATA}.png"); plt.close()

        # per-kernel metrics + bars (same as v7; method-suffixed CSV)
        def _safe_mean(x): return float(np.mean(x)) if len(x) else float("nan")
        rows=[]
        forecast_rmse={}; areg_rmse={}; recall={}; res_std_raw={};res_std_norm={};res_std_acc={}
        for k in ds.kernel_names:
            forecast_rmse[k] = math.sqrt(_safe_mean(forecast_mse_by_kernel[k]))
            areg_rmse[k]     = math.sqrt(_safe_mean(areg_mse_by_kernel[k]))
            sup=cls_support_by_kernel[k]; tp=cls_tp_by_kernel[k]; recall[k]=(tp/sup) if sup>0 else float("nan")
            def _concat_std(buckets):
                if len(buckets.get(k,[]))==0: return float("nan")
                R=np.concatenate(buckets[k],axis=0).ravel(); return float(np.std(R))
            res_std_raw[k]=_concat_std(residuals_by_kernel); res_std_norm[k]=_concat_std(residuals_norm_by_kernel); res_std_acc[k]=_concat_std(residuals_acc_by_kernel)
            rows.append({"kernel":k,"forecast_RMSE":forecast_rmse[k],"classification_recall":recall[k],
                         "regression_RMSE":areg_rmse[k],"residual_std_raw":res_std_raw[k],
                         "residual_std_norm":res_std_norm[k],"residual_std_acc":res_std_acc[k]})
        df_perk=pd.DataFrame(rows); ensure_dir(ML_ROOT / "analysis")
        df_perk.to_csv(ML_ROOT / "analysis" / f"per_kernel_metrics__{METHOD_FOR_DATA}.csv", index=False)

        def _bar(metric_dict,title,ylabel,filename):
            names=list(metric_dict.keys()); vals=[metric_dict[n] for n in names]
            plt.figure(figsize=(10,4)); plt.bar(range(len(names)), vals)
            plt.xticks(range(len(names)), names, rotation=30, ha="right"); plt.title(title); plt.ylabel(ylabel)
            plt.tight_layout(); plt.savefig(ML_ROOT / f"{filename}__{METHOD_FOR_DATA}.png"); plt.close()
        _bar(forecast_rmse,"Forecast RMSE per Kernel","RMSE","bars_forecast_rmse")
        _bar(recall,"Classification Recall per Kernel","Recall","bars_per_kernel_recall")
        _bar(areg_rmse,"Amplitude Regression RMSE per Kernel","RMSE","bars_regression_rmse")
        _bar(res_std_acc,"Physics Residual Std per Kernel (acc)","Std (acc-based)","bars_residual_std_acc")
        _bar(res_std_norm,"Physics Residual Std per Kernel (normalized)","Std (normalized)","bars_residual_std_norm")
        _bar(res_std_raw,"Physics Residual Std per Kernel (raw)","Std (raw)","bars_residual_std_raw")

        # your per-kernel residual hist+bars helper (unchanged)
        def _per_kernel_hist_and_bars(by_k, tag, out_bar):
            means, stds, names = [], [], []
            by_k = {k: v for k, v in by_k.items() if len(v) > 0}
            for kname, items in by_k.items():
                R = np.concatenate(items, axis=0).ravel()
                means.append(np.mean(R)); stds.append(np.std(R)); names.append(kname)
                plt.figure(); plt.hist(R, bins=50, alpha=0.85)
                plt.title(f"Residuals histogram — {kname} ({tag})")
                plt.tight_layout(); plt.savefig(ML_ROOT / "samples" / f"residual_histogram__{kname}__{tag}__{METHOD_FOR_DATA}.png"); plt.close()
            if names:
                x = np.arange(len(names))
                plt.figure(figsize=(8,4))
                plt.bar(x, means, yerr=stds, capsize=4)
                plt.xticks(x, names, rotation=30, ha="right")
                plt.ylabel(f"Residual {tag} mean ± std")
                plt.title(f"Physics residuals per kernel family ({tag}) — {METHOD_FOR_DATA}")
                plt.tight_layout(); plt.savefig(ML_ROOT / "samples" / f"{out_bar}__{METHOD_FOR_DATA}.png"); plt.close()

        _per_kernel_hist_and_bars(residuals_by_kernel, "raw", "residual_summary_bars")
        _per_kernel_hist_and_bars(residuals_norm_by_kernel, "normalized", "residual_summary_bars_norm")
        _per_kernel_hist_and_bars(residuals_acc_by_kernel, "acc", "residual_summary_bars_acc")

        # overall metrics (NPSSL)
        reg_mae=mean_absolute_error(a_true_all.ravel(), a_pred_all.ravel())
        reg_rmse=mean_squared_error(a_true_all.ravel(), a_pred_all.ravel(), squared=False)
        reg_r2=r2_score(a_true_all.ravel(), a_pred_all.ravel())
        fore_mae=mean_absolute_error(y_true_all.ravel(), y_pred_all.ravel())
        fore_rmse=mean_squared_error(y_true_all.ravel(), y_pred_all.ravel(), squared=False)
        fore_r2=r2_score(y_true_all.ravel(), y_pred_all.ravel())
        res_mean_raw=float(np.mean(residuals_all)); res_std_raw=float(np.std(residuals_all))
        res_mean_norm=float(np.mean(residuals_norm_all)); res_std_norm=float(np.std(residuals_norm_all))
        res_mean_acc=float(np.mean(residuals_acc_all)); res_std_acc=float(np.std(residuals_acc_all))

        metrics = {
            "method": METHOD_FOR_DATA,
            "model": "NPSSL",
            "classification_accuracy": float(acc),
            "regression_MAE": float(reg_mae),
            "regression_RMSE": float(reg_rmse),
            "regression_R2": float(reg_r2),
            "forecast_MAE": float(fore_mae),
            "forecast_RMSE": float(fore_rmse),
            "forecast_R2": float(fore_r2),
            "residual_mean": res_mean_raw, "residual_std": res_std_raw,
            "residual_norm_mean": res_mean_norm, "residual_norm_std": res_std_norm,
            "residual_acc_mean": res_mean_acc, "residual_acc_std": res_std_acc,
        }
        with open(ML_ROOT / f"metrics__{METHOD_FOR_DATA}.json", "w", encoding="utf-8") as fjson:
            json.dump(metrics, fjson, indent=2)

        # ================================
        # Baselines (GRU + Transformer)
        # ================================
        def train_eval_baseline(model_name: str):
            # dirs
            base_dir = ML_ROOT / "baselines" / model_name / METHOD_FOR_DATA
            ensure_dir(base_dir); ensure_dir(base_dir / "arrays")

            # model choice
            if model_name == "GRU":
                mdl = GRUForecaster(in_feats=5, hidden=HIDDEN, horizon=HORIZON).to(device)
                lr_local = LR
            else:
                mdl = TinyTransformerForecaster(
                    in_feats=5, d_model=64, nhead=4, num_layers=2, horizon=HORIZON, dim_ff=128
                ).to(device)
                # Transformers are more sensitive → use smaller LR
                lr_local = min(LR, 1e-4)

            optb = torch.optim.Adam(mdl.parameters(), lr=lr_local, weight_decay=WEIGHT_DECAY)
            schedb = torch.optim.lr_scheduler.ReduceLROnPlateau(optb, factor=0.5, patience=3)

            def _epoch(dl, train=True):
                if train: mdl.train()
                else: mdl.eval()
                tot = 0.0; ns = 0
                with torch.set_grad_enabled(train):
                    for batch in dl:
                        X, yf = batch["X"].to(device), batch["y_fore"].to(device)

                        # normalize inputs (per sequence)
                        X = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-8)

                        yhat = mdl(X)
                        loss = F.mse_loss(yhat, yf)

                        # guard against NaNs
                        if not torch.isfinite(loss):
                            continue

                        if train:
                            optb.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                            optb.step()

                        ns += X.size(0)
                        tot += float(loss) * X.size(0)

                return (tot / ns) if ns > 0 else float("nan")

            # choose a non-empty validation loader (fallback to train if val empty)
            dl_val_eff = dl_va if len(dl_va) > 0 else dl_tr

            best = float("inf")
            best_file = base_dir / "best.pt"

            for ep in range(1, EPOCHS + 1):
                trl = _epoch(dl_tr, True)
                val = _epoch(dl_val_eff, False)

                if not np.isfinite(val):
                    val = 1e9
                schedb.step(val)

                # always save at first epoch or when improved or if file missing
                if (ep == 1) or (val < best) or (not best_file.exists()):
                    best = val
                    torch.save(mdl.state_dict(), best_file)

                print(f"[{METHOD_FOR_DATA}][{model_name}] epoch {ep:02d} train={trl:.4f} val={(val if np.isfinite(val) else float('nan')):.4f}")

            if not best_file.exists():
                torch.save(mdl.state_dict(), best_file)
            mdl.load_state_dict(torch.load(best_file, map_location=device))
            mdl.eval()

            # test
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch in dl_te:
                    X, yf = batch["X"].to(device), batch["y_fore"].to(device)
                    X = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-8)
                    yhat = mdl(X)
                    y_true.append(yf.cpu().numpy())
                    y_pred.append(yhat.cpu().numpy())

            y_true = np.concatenate(y_true, axis=0)
            y_pred = np.concatenate(y_pred, axis=0)

            # sanitize predictions to avoid NaNs/Infs
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

            np.save(base_dir / "arrays" / "y_true_forecast.npy", y_true)
            np.save(base_dir / "arrays" / "y_pred_forecast.npy", y_pred)

            mae  = mean_absolute_error(y_true.ravel(), y_pred.ravel())
            rmse = mean_squared_error(y_true.ravel(), y_pred.ravel(), squared=False)
            r2   = r2_score(y_true.ravel(), y_pred.ravel())

            # quick plot
            plt.figure()
            sl = min(200, y_true.shape[0] * y_true.shape[1])
            plt.plot(y_true.ravel()[:sl], label="True", alpha=0.7)
            plt.plot(y_pred.ravel()[:sl], label="Pred", alpha=0.7)
            plt.legend()
            plt.title(f"{model_name} forecast (slice) — {METHOD_FOR_DATA}")
            plt.tight_layout()
            plt.savefig(base_dir / "forecast_vs_true.png")
            plt.close()

            return {
                "method": METHOD_FOR_DATA, "model": model_name,
                "classification_accuracy": np.nan,
                "regression_MAE": np.nan, "regression_RMSE": np.nan, "regression_R2": np.nan,
                "forecast_MAE": float(mae), "forecast_RMSE": float(rmse), "forecast_R2": float(r2),
                "residual_mean": np.nan, "residual_std": np.nan,
                "residual_norm_mean": np.nan, "residual_norm_std": np.nan,
                "residual_acc_mean": np.nan, "residual_acc_std": np.nan
            }


        # train/eval both baselines and aggregate rows
        row_np = metrics
        row_gru = train_eval_baseline("GRU")
        row_trf = train_eval_baseline("Transformer")
        rows_all_methods.extend([row_np, row_gru, row_trf])

    # ========================
    # Write comparison tables
    # ========================
    df_all = pd.DataFrame(rows_all_methods)
    df_all.to_csv(ML_ROOT / "analysis" / "per_method_metrics.csv", index=False)
    # A condensed SOTA-like CSV (use in your paper table)
    cols = ["method","model","forecast_RMSE","classification_accuracy","regression_RMSE","residual_norm_std"]
    df_all[cols].to_csv(ML_ROOT / "analysis" / "sota_like_table.csv", index=False)

    print("\n[v8] Done. NPSSL + baselines trained/evaluated across all numerical methods; analysis CSVs saved.")


if __name__ == "__main__":
    run_experiment()

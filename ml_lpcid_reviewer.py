#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_solver_v8.py (LPCID real-data edition)
-----------------------------------------
Evaluates the NPSSL pipeline on real LPCID free-vibration data sitting in:
    ./datasets/LPCID/prepped/*.csv
Each CSV must have columns: time, f, u, v, a, H_hat
We map them to the internal convention: [forcing, disp, vel, acc, hist].

What stays the same:
  • Model: Causal TCN encoder + NPSSL + heads
  • Physics residual, omega estimation, curriculum, logging, exports
  • Windowing defaults (W=512, H=64, stride=32)
  • Export structure (metrics, arrays, plots)

What’s different for real data:
  • No class labels, no ground-truth amplitudes → set their loss weights to 0.0
  • Single “method” slot called "lpcid" (no synthetic solver variants)
  • Confusion-matrix and per-kernel tables are skipped
  • Output root: ./results/ml_v8_lpcid

This file is adapted to your v8 pipeline while redirecting the dataset to LPCID.
"""

from __future__ import annotations
import os, json, math, pathlib, random, sys, glob, re
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------
# IO LOCATIONS (hardcoded)
# --------------------------
LPCID_DIR   = pathlib.Path("./datasets/LPCID/prepped").resolve()
ML_ROOT     = pathlib.Path("./results/ml_v8").resolve()

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
FS_FALLBACK        = 500.0     # Hz, if time-step inference fails
DT_FALLBACK        = 1.0 / FS_FALLBACK

# Windowing
WINDOW             = 512
HORIZON            = 64
STRIDE             = 32

# Model sizes
HIDDEN             = 128
PRONY_M            = 24
TCN_CHANNELS       = 64

# Optimization
EPOCHS             = 30
BATCH_SIZE         = 64
LR                 = 1e-3
WEIGHT_DECAY       = 1e-5

# Loss weights (REAL DATA: disable class & amplitude heads)
L_FORECAST         = 1.0
L_CLASS            = 0.0      # <- no labels in LPCID
L_A_REG            = 0.0      # <- no ground-truth amplitudes
L_PHYS_TARGET      = 0.9

# Physics curriculum
PHYS_WARMUP_EPOCHS = 8
L_PHYS_WARMUP      = 0.1

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

def _moving_avg_kernel(win: int) -> torch.Tensor:
    k = torch.ones(win, dtype=torch.float32) / float(win)
    return k.view(1, 1, -1)

def _smooth_1d(x: torch.Tensor, win: int = 9) -> torch.Tensor:
    if win <= 1:
        return x
    pad = (win - 1, 0)
    k = _moving_avg_kernel(win).to(x.device)
    x_ = F.pad(x.unsqueeze(1), pad=(pad[0], pad[1]))
    y = F.conv1d(x_, k).squeeze(1)
    return y

def _second_derivative(x: torch.Tensor, dt: float) -> torch.Tensor:
    coeff = torch.tensor([-1, 12, -39, 56, -39, 12, -1], dtype=x.dtype, device=x.device) / (6.0 * dt * dt)
    coeff = coeff.view(1, 1, -1)
    xx = x.unsqueeze(1)
    u2_full = torch.nn.functional.conv1d(xx, coeff, padding=0)
    return u2_full.squeeze(1)

def _parabolic_interp(mag_row, idx):
    a = mag_row[idx-1]; b = mag_row[idx]; c = mag_row[idx+1]
    denom = (a - 2*b + c)
    if denom.abs() < 1e-12: return 0.0
    return 0.5 * (a - c) / denom

def _estimate_omega_batch(u: torch.Tensor, dt: float, k: int = 2) -> torch.Tensor:
    U = torch.fft.rfft(u, dim=1)
    mag = torch.abs(U)
    mag[:, 0] = 0.0
    vals, idxs = torch.topk(mag, k=max(1, k), dim=1)
    peak_idx = idxs[:, min(k-1, idxs.size(1)-1)]
    peak_idx = torch.clamp(peak_idx, 1, mag.size(1)-2)
    offsets = []
    for i in range(mag.size(0)):
        offsets.append(_parabolic_interp(mag[i], int(peak_idx[i])) )
    delta = torch.tensor(offsets, device=mag.device, dtype=mag.dtype)
    bin_width = (1.0/dt) / (2*(mag.size(1)-1))
    f_refined = peak_idx * bin_width + delta * bin_width
    omega = 2.0 * torch.pi * f_refined
    return omega.view(-1, 1)

# --------------------------
# DATASET (LPCID real data)
# --------------------------
def _infer_dt(time_col: np.ndarray) -> float:
    if time_col is None or len(time_col) < 2:
        return DT_FALLBACK
    d = np.diff(time_col)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return DT_FALLBACK
    dt = float(np.median(d))
    return dt if dt > 0 else DT_FALLBACK

def _load_one_csv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected columns: time,f,u,v,a,H_hat
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("time",): rename[c] = "time"
        elif cl in ("f","force","forcing"): rename[c] = "forcing"
        elif cl in ("u","disp","displacement","x"): rename[c] = "disp"
        elif cl in ("v","vel","velocity"): rename[c] = "vel"
        elif cl in ("a","acc","accel","acceleration"): rename[c] = "acc"
        elif cl in ("h_hat","hhat","h","hist","hereditary"): rename[c] = "hist"
    df = df.rename(columns=rename)
    needed = {"time","forcing","disp","vel","acc","hist"}
    if not needed.issubset(set(df.columns)):
        missing = needed - set(df.columns)
        raise ValueError(f"{path.name}: missing columns {missing}")
    # ensure numeric
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(needed))
    return df[["time","forcing","disp","vel","acc","hist"]].reset_index(drop=True)

def _stack_all_lpcid(directory: pathlib.Path) -> Tuple[pd.DataFrame, float]:
    files = sorted(glob.glob(str(directory / "*__phys.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files under {directory}")
    parts = []
    dts = []
    for f in files:
        dfi = _load_one_csv(pathlib.Path(f))
        dt = _infer_dt(dfi["time"].to_numpy())
        dts.append(dt)
        # keep as is; we concatenate with a separator row to prevent windowing across files
        parts.append(dfi)
        # add tiny gap (NaNs) to block cross-file windows
        parts.append(pd.DataFrame({"time":[np.nan],"forcing":[np.nan],"disp":[np.nan],"vel":[np.nan],"acc":[np.nan],"hist":[np.nan]}))
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    dt_global = float(np.median(dts)) if dts else DT_FALLBACK
    return df, dt_global

class LPCIDWindowedDataset(Dataset):
    """
    Windows from real LPCID streams (no class labels, no amplitude ground-truth).
    Returns dict with X=[f,u,v,a,H], y_fore=future u. Dummy fields are provided
    for compatibility (class id = 0; amplitudes = zeros; mask = zeros).
    """
    def __init__(self, root: pathlib.Path, window: int, horizon: int, stride: int):
        super().__init__()
        self.window = window
        self.horizon = horizon
        self.stride = stride
        df, dt = _stack_all_lpcid(root)
        self.df = df
        self.dt = dt
        # materialize window start indices
        N = len(self.df)
        self.starts = list(range(0, N - (window + horizon) + 1, stride))

    def __len__(self): return len(self.starts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        s = self.starts[i]
        w, h = self.window, self.horizon
        seg_in  = self.df.iloc[s:s+w]
        seg_out = self.df.iloc[s+w:s+w+h]
        X = np.stack([
            seg_in["forcing"].to_numpy(np.float32),
            seg_in["disp"].to_numpy(np.float32),
            seg_in["vel"].to_numpy(np.float32),
            seg_in["acc"].to_numpy(np.float32),
            seg_in["hist"].to_numpy(np.float32),
        ], axis=-1)
        y_fore = seg_out["disp"].to_numpy(np.float32)
        # dummy compatibility
        y_cls  = 0
        a_true = np.zeros((PRONY_M,), dtype=np.float32)
        mask_a = np.zeros((PRONY_M,), dtype=np.float32)
        return {"X":X, "y_fore":y_fore, "y_cls":y_cls, "a_true":a_true, "mask_a":mask_a}

def train_val_test_split_real(ds: Dataset):
    n = len(ds)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)

    # Default splits
    n_tr = max(1, int(0.7 * n))
    n_va = max(1, int(0.15 * n))
    if n_tr + n_va >= n:
        n_va = 1
        n_tr = max(1, n - 2)
    n_te = max(1, n - (n_tr + n_va))

    tr_idx = idx[:n_tr]
    va_idx = idx[n_tr:n_tr+n_va]
    te_idx = idx[n_tr+n_va:n_tr+n_va+n_te]

    class _Subset(Dataset):
        def __init__(self, base: Dataset, indices: np.ndarray):
            self.base = base; self.indices = indices
        def __len__(self): return len(self.indices)
        def __getitem__(self, i): return self.base[int(self.indices[i])]

    return _Subset(ds, tr_idx), _Subset(ds, va_idx), _Subset(ds, te_idx)


# --------------------------
# MODEL: Causal TCN + NPSSL
# --------------------------
class CausalTCN(nn.Module):
    def __init__(self, in_ch=5, channels=64):
        super().__init__()
        pad1, dil1 = 2, 2
        pad2, dil2 = 4, 4
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad1, dilation=dil1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad2, dilation=dil2),
            nn.GELU(),
        )
    def forward(self, x):  # [B,W,F]
        x = x.transpose(1, 2)
        y = self.net(x)
        return y.transpose(1, 2)

class NPSSL(nn.Module):
    def __init__(self, prony_M: int, dt: float, tmin=None, tmax=None):
        super().__init__()
        self.M = prony_M
        self.dt = dt
        tmin = 5*dt if tmin is None else tmin
        tmax = 10.0 if tmax is None else tmax
        b = np.geomspace(1.0/tmax, 1.0/max(tmin, 1e-4), num=prony_M).astype(np.float32)
        self.register_buffer("b_grid", torch.from_numpy(b))
        self.a_raw = nn.Parameter(torch.full((prony_M,), -2.0))
        self.softplus = nn.Softplus()

    def forward(self, v_seq: torch.Tensor):
        B, W = v_seq.shape
        a = self.softplus(self.a_raw)                 # [M] >= 0
        decay = torch.exp(-self.b_grid * self.dt)     # [M]
        z = torch.zeros((B, self.M), device=v_seq.device)
        H = []
        for n in range(W):
            z = z * decay + a * (self.dt * v_seq[:, n:n+1])
            H.append(z.sum(dim=1, keepdim=True))
        H_seq = torch.cat(H, dim=1)                  # [B, W]
        a_pred = a.unsqueeze(0).repeat(B, 1)         # [B,M] (constant across time)
        return H_seq, a_pred

class MultiTaskModel(nn.Module):
    def __init__(self, n_features: int, horizon: int, n_classes: int, prony_M: int, dt: float, hidden: int = HIDDEN, tcn_channels: int = TCN_CHANNELS):
        super().__init__()
        self.horizon = horizon
        self.dt = dt
        self.enc = CausalTCN(in_ch=n_features, channels=tcn_channels)
        self.npssl = NPSSL(prony_M=prony_M, dt=dt)
        self.head_fore = nn.Sequential(nn.Linear(tcn_channels, hidden), nn.GELU(), nn.Linear(hidden, horizon))
        self.head_cls  = nn.Sequential(nn.Linear(tcn_channels, hidden), nn.GELU(), nn.Linear(hidden, n_classes))
        self.head_amp  = nn.Sequential(nn.Linear(tcn_channels, hidden), nn.GELU(), nn.Linear(hidden, prony_M), nn.Softplus())

    def forward(self, X: torch.Tensor, dt: float, omega) -> dict:
        B, W, C = X.shape
        device = X.device
        forcing = X[:, :, 0]
        u       = X[:, :, 1]
        v       = X[:, :, 2]

        feats  = self.enc(X)
        pooled = feats[:, -1, :]
        y_fore = self.head_fore(pooled)
        y_cls  = self.head_cls(pooled)
        a_pred = self.head_amp(pooled)

        H_hat, _ = self.npssl(v)
        u_s    = _smooth_1d(u, win=9)
        u2     = _second_derivative(u_s, dt=dt)
        H_crop = H_hat[:, 3:-3]
        u_crop = u[:, 3:-3]
        f_crop = forcing[:, 3:-3]

        if isinstance(omega, (float, int)):
            omega_b = torch.full((B, 1), float(omega), device=device, dtype=u.dtype)
        else:
            omega_t = torch.as_tensor(omega, device=device, dtype=u.dtype)
            omega_b = omega_t.view(-1, 1) if omega_t.ndim == 1 else omega_t

        residual = u2 + (omega_b**2) * u_crop + H_crop - f_crop
        return {"y_fore": y_fore, "y_cls": y_cls, "a_pred": a_pred, "residual": residual}

# --------------------------
# MAIN TRAIN/TEST
# --------------------------
def run_experiment():
    set_all_seeds(SEED)
    global WINDOW, HORIZON, STRIDE
    # dirs & tee log
    ensure_dir(ML_ROOT); ensure_dir(ML_ROOT / "checkpoints")
    ensure_dir(ML_ROOT / "arrays"); ensure_dir(ML_ROOT / "samples"); ensure_dir(ML_ROOT / "analysis")
    _log_f = open(ML_ROOT / "output.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_f)
    sys.stderr = _Tee(sys.stderr, _log_f)

    # dataset
    print(f"[LPCID] Loading real data from: {LPCID_DIR}")
    ds = LPCIDWindowedDataset(LPCID_DIR, WINDOW, HORIZON, STRIDE)

    # Make window/horizon adaptive if dataset is tiny or fs is low
    if len(ds) < 32 or ds.dt > 0.2:  # very low fs or few windows
        # ~15s window, ~4s horizon at current dt; clamp to sensible bounds
        WINDOW  = int(max(32, min(256, round(15.0 / ds.dt))))
        HORIZON = int(max(8,  min(64,  round(4.0  / ds.dt))))
        STRIDE  = max(4, WINDOW // 4)
        print(f"[LPCID] Adapted windowing: W={WINDOW}, H={HORIZON}, stride={STRIDE}")
        # Rebuild dataset with new windowing
        ds = LPCIDWindowedDataset(LPCID_DIR, WINDOW, HORIZON, STRIDE)
        print(f"[LPCID] windows after adaptation: {len(ds)}")


    dt = ds.dt
    print(f"[LPCID] dt ≈ {dt:.6f} s (fs ≈ {1.0/dt:.2f} Hz), windows={len(ds)}")

    # splits & loaders
    tr, va, te = train_val_test_split_real(ds)

    # Dynamic, small batch size so we never drop everything
    dyn_bs = max(1, min(16, len(tr)))
    dl_tr = DataLoader(tr, batch_size=dyn_bs, shuffle=True,  drop_last=False)
    dl_va = DataLoader(va, batch_size=max(1, min(16, len(va))), shuffle=False, drop_last=False)
    dl_te = DataLoader(te, batch_size=max(1, min(16, len(te))), shuffle=False, drop_last=False)

    print(f"[LPCID] batch sizes -> train:{dyn_bs}, val:{max(1, min(16, len(va)))}, test:{max(1, min(16, len(te)))}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(n_features=5, horizon=HORIZON, n_classes=1, prony_M=PRONY_M, dt=dt,
                           hidden=HIDDEN, tcn_channels=TCN_CHANNELS).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    # train
    log_rows = []; best_va_core = float("inf")
    best_path = ML_ROOT / "checkpoints" / "best_model.pt"
    for epoch in range(1, EPOCHS + 1):
        L_PHYS = L_PHYS_WARMUP if epoch <= PHYS_WARMUP_EPOCHS else (
            L_PHYS_WARMUP + min(1.0, (epoch - PHYS_WARMUP_EPOCHS)/max(1, EPOCHS - PHYS_WARMUP_EPOCHS)) * (L_PHYS_TARGET - L_PHYS_WARMUP)
        )
        w_cls = L_CLASS   # 0.0
        w_a   = L_A_REG   # 0.0

        # train epoch
        model.train(); tr_loss=tr_fore=tr_phys=0.0; seen=0
        for batch in dl_tr:
            X  = torch.as_tensor(batch["X"], dtype=torch.float32, device=device)
            yf = torch.as_tensor(batch["y_fore"], dtype=torch.float32, device=device)

            omega_b = _estimate_omega_batch(X[:, :, 1], dt, k=2)
            out = model(X, dt, omega_b)

            loss_fore = ((out["y_fore"]-yf)**2).mean()
            # physics residual normalization (as in v8)
            forcing,u,v = X[:,:,0],X[:,:,1],X[:,:,2]
            u2 = _second_derivative(_smooth_1d(u,win=9), dt=dt)
            H_hat,_ = model.npssl(v); Hc=H_hat[:,3:-3]; uc=u[:,3:-3]; fc=forcing[:,3:-3]
            denom = (torch.abs(u2)+torch.abs((omega_b**2)*uc)+torch.abs(Hc)+torch.abs(fc)).mean(dim=1,keepdim=True)+1e-8
            r_norm = out["residual"]/denom
            loss_phys = F.smooth_l1_loss(r_norm, torch.zeros_like(r_norm))

            loss = (L_FORECAST*loss_fore) + (L_PHYS*loss_phys)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bsz=X.size(0); seen+=bsz
            tr_loss+=float(loss)*bsz; tr_fore+=float(loss_fore)*bsz; tr_phys+=float(loss_phys)*bsz

        if seen > 0:
            tr_loss/=seen; tr_fore/=seen; tr_phys/=seen
        else:
            tr_loss = tr_fore = tr_phys = float('nan')

        # val
        model.eval(); va_loss=va_fore=va_phys=0.0; seen=0
        with torch.no_grad():
            for batch in dl_va:
                X  = torch.as_tensor(batch["X"], dtype=torch.float32, device=device)
                yf = torch.as_tensor(batch["y_fore"], dtype=torch.float32, device=device)
                omega_b = _estimate_omega_batch(X[:, :, 1], dt, k=2)
                out = model(X, dt, omega_b)
                loss_fore = ((out["y_fore"]-yf)**2).mean()
                forcing,u,v = X[:,:,0],X[:,:,1],X[:,:,2]
                u2 = _second_derivative(_smooth_1d(u,win=9), dt=dt)
                H_hat,_ = model.npssl(v); Hc=H_hat[:,3:-3]; uc=u[:,3:-3]; fc=forcing[:,3:-3]
                denom=(torch.abs(u2)+torch.abs((omega_b**2)*uc)+torch.abs(Hc)+torch.abs(fc)).mean(dim=1,keepdim=True)+1e-8
                r_norm = out["residual"]/denom
                loss_phys = F.smooth_l1_loss(r_norm, torch.zeros_like(r_norm))
                loss = (L_FORECAST*loss_fore) + (L_PHYS*loss_phys)

                bsz=X.size(0); seen+=bsz
                va_loss+=float(loss)*bsz; va_fore+=float(loss_fore)*bsz; va_phys+=float(loss_phys)*bsz
        if seen > 0:
            va_loss/=seen; va_fore/=seen; va_phys/=seen
        else:
            va_loss = va_fore = va_phys = float('nan')
            
        va_core = (L_FORECAST*va_fore)  # core excludes physics for scheduler (as in v8 spirit)
        sched.step(va_core)

        if va_core < best_va_core:
            best_va_core = va_core
            ensure_dir(best_path.parent)
            torch.save({"model":model.state_dict(),
                        "config":{"WINDOW":WINDOW,"HORIZON":HORIZON,"PRONY_M":PRONY_M,"dt":dt}},
                       best_path)

        log_rows.append({"epoch":epoch,"train_loss":tr_loss,"val_loss":va_loss,"val_core":va_core,
                         "tr_fore":tr_fore,"tr_phys":tr_phys,"va_fore":va_fore,"va_phys":va_phys,
                         "L_PHYS":L_PHYS})
        print(f"[lpcid] epoch {epoch:02d} train={tr_loss:.4f} val={va_loss:.4f} core={va_core:.4f}")

    

    # log curves
    df_log = pd.DataFrame(log_rows); ensure_dir(ML_ROOT / "logs")
    df_log.to_csv(ML_ROOT / "logs" / "training_log__lpcid.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(df_log["epoch"], df_log["train_loss"], label="Training loss", linewidth=2.5)
    plt.plot(df_log["epoch"], df_log["val_loss"], label="Validation loss", linewidth=2.5)
    plt.plot(df_log["epoch"], df_log["val_core"], label="Validation forecast loss", linestyle="--", linewidth=2.5)
    plt.xlabel("Epoch", fontsize=18, fontweight="bold")
    plt.ylabel("Loss value", fontsize=18, fontweight="bold")
    plt.title("LPCID training and validation loss", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=14, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ML_ROOT / "loss_curves__lpcid.png", dpi=600, bbox_inches="tight")
    plt.close()

    # test
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    model.eval()

    y_true_all=[]; y_pred_all=[]
    residuals_all=[]; residuals_norm_all=[]; residuals_acc_all=[]
    with torch.no_grad():
        for batch in dl_te:
            X  = torch.as_tensor(batch["X"], dtype=torch.float32, device=device)
            yf = torch.as_tensor(batch["y_fore"], dtype=torch.float32, device=device)
            omega_b = _estimate_omega_batch(X[:, :, 1], dt, k=2)
            out = model(X, dt, omega_b)

            y_true_all.append(yf.cpu().numpy())
            y_pred_all.append(out["y_fore"].cpu().numpy())

            res_np = out["residual"].cpu().numpy(); residuals_all.append(res_np)

            forcing,u,v = X[:,:,0],X[:,:,1],X[:,:,2]
            u2 = _second_derivative(_smooth_1d(u,win=9), dt=dt)
            H_hat,_ = model.npssl(v); Hc=H_hat[:,3:-3]; uc=u[:,3:-3]; fc=forcing[:,3:-3]
            denom=(torch.abs(u2)+torch.abs((omega_b**2)*uc)+torch.abs(Hc)+torch.abs(fc)).mean(dim=1,keepdim=True)+1e-8
            r_norm = (out["residual"]/denom).cpu().numpy(); residuals_norm_all.append(r_norm)
            # "acc-based" residual: a + ω^2 u + H - f  (a is measured channel)
            a_meas = X[:,:,3]; ac=a_meas[:,3:-3]; r_acc=(ac+(omega_b**2)*uc+Hc-fc).cpu().numpy()
            residuals_acc_all.append(r_acc)

    y_true_all=np.concatenate(y_true_all,axis=0); y_pred_all=np.concatenate(y_pred_all,axis=0)
    residuals_all=np.concatenate(residuals_all,axis=0)
    residuals_norm_all=np.concatenate(residuals_norm_all,axis=0)
    residuals_acc_all=np.concatenate(residuals_acc_all,axis=0)

    # save arrays
    arr_dir = ML_ROOT / "arrays"; ensure_dir(arr_dir)
    np.save(arr_dir / "y_true_forecast__lpcid.npy", y_true_all)
    np.save(arr_dir / "y_pred_forecast__lpcid.npy", y_pred_all)
    np.save(arr_dir / "residuals__lpcid.npy", residuals_all)
    np.save(arr_dir / "residuals_normalized__lpcid.npy", residuals_norm_all)
    np.save(arr_dir / "residuals_acc__lpcid.npy", residuals_acc_all)

    # quick plots
    plt.figure(figsize=(10, 6))
    sl=min(200, y_true_all.shape[0]*y_true_all.shape[1])
    sample_index = np.arange(sl)
    time_axis = sample_index * dt
    plt.plot(time_axis, y_true_all.ravel()[:sl], label="True displacement", alpha=0.85, linewidth=2.5)
    plt.plot(time_axis, y_pred_all.ravel()[:sl], label="Predicted displacement", alpha=0.85, linewidth=2.5)
    plt.xlabel("Time within forecast slice (s)", fontsize=18, fontweight="bold")
    plt.ylabel("Displacement amplitude", fontsize=18, fontweight="bold")
    plt.title("LPCID displacement forecast versus ground truth", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=14, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ML_ROOT / "forecast_vs_true__lpcid.png", dpi=600, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals_all.ravel(), bins=50, alpha=0.85)
    plt.xlabel("Raw physics residual", fontsize=18, fontweight="bold")
    plt.ylabel("Frequency", fontsize=18, fontweight="bold")
    plt.title("LPCID raw physics residual distribution", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ML_ROOT / "residual_hist__lpcid.png", dpi=600, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals_norm_all.ravel(), bins=50, alpha=0.85)
    plt.xlabel("Normalized physics residual, $\tilde{r}$", fontsize=18, fontweight="bold")
    plt.ylabel("Frequency", fontsize=18, fontweight="bold")
    plt.title("LPCID normalized physics residual distribution", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ML_ROOT / "residual_hist_norm__lpcid.png", dpi=600, bbox_inches="tight")
    plt.close()

    # overall metrics
    fore_mae  = mean_absolute_error(y_true_all.ravel(), y_pred_all.ravel())
    fore_rmse = mean_squared_error(y_true_all.ravel(), y_pred_all.ravel(), squared=False)
    fore_r2   = r2_score(y_true_all.ravel(), y_pred_all.ravel())
    res_mean_raw  = float(np.mean(residuals_all));      res_std_raw  = float(np.std(residuals_all))
    res_mean_norm = float(np.mean(residuals_norm_all)); res_std_norm = float(np.std(residuals_norm_all))
    res_mean_acc  = float(np.mean(residuals_acc_all));  res_std_acc  = float(np.std(residuals_acc_all))

    metrics = {
        "method": "lpcid",
        "model": "NPSSL",
        "classification_accuracy": None,   # not applicable
        "regression_MAE": None, "regression_RMSE": None, "regression_R2": None,
        "forecast_MAE": float(fore_mae),
        "forecast_RMSE": float(fore_rmse),
        "forecast_R2": float(fore_r2),
        "residual_mean": res_mean_raw, "residual_std": res_std_raw,
        "residual_norm_mean": res_mean_norm, "residual_norm_std": res_std_norm,
        "residual_acc_mean": res_mean_acc, "residual_acc_std": res_std_acc,
    }
    with open(ML_ROOT / "metrics__lpcid.json", "w", encoding="utf-8") as fjson:
        json.dump(metrics, fjson, indent=2)

    # condensed table for your paper
    df = pd.DataFrame([{
        "method":"lpcid","model":"NPSSL",
        "forecast_RMSE":metrics["forecast_RMSE"],
        "classification_accuracy":np.nan,
        "regression_RMSE":np.nan,
        "residual_norm_std":metrics["residual_norm_std"]
    }])
    df.to_csv(ML_ROOT / "analysis" / "sota_like_table__lpcid.csv", index=False)

    print("\n[LPCID] Done. NPSSL trained/evaluated on real data; artifacts saved under results/ml_v8_lpcid.")

if __name__ == "__main__":
    run_experiment()

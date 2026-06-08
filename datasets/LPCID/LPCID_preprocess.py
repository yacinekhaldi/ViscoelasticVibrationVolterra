# prep_lpcid_to_npSSL_hardcoded.py
# --------------------------------
# Preprocess LPCID free-vibration Excel files into NPSSL-ready series:
#   columns: time, f, u, v, a, H_hat
# Optional: export sliding windows X=[f,u,v,a,H_hat], Y=future u
#
# How to use:
#   python prep_lpcid_to_npSSL_hardcoded.py
#
# Tune the constants in the CONFIG section below.

import os, re, glob
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq

# ----------------------------- CONFIG ------------------------------------
INPUT_DIR   = "."                # folder with LPCID Excel files
OUTPUT_DIR  = "./prepped"        # where to write CSV/NPZ
MAKE_WINDOWS = True              # also export training windows
W           = 512                # window length (samples)
H_FOR       = 64                 # forecast horizon (samples)
STRIDE      = 128                # stride between windows

# Physics / smoothing
MASS        = 1.20               # kg (set to your rig mass); if None -> 1.0
STIFFNESS   = None               # N/m; if None -> estimate via dominant freq of u(t)
FMIN_PEAK   = 0.2                # Hz, min freq in peak search (ignore DC)

# Savitzky–Golay for derivatives (use odd window >= poly+2)
SG_WINDOW   = 31                 # samples
SG_POLY     = 3

# -------------------------------------------------------------------------

CANDIDATE_TIME  = re.compile(r'^(time|t)\b', re.I)
CANDIDATE_DISP  = re.compile(r'(disp|displ|x\b|u\b)', re.I)
CANDIDATE_VEL   = re.compile(r'(vel|velocity|v\b)', re.I)
CANDIDATE_ACC   = re.compile(r'(acc|accel|a\b)', re.I)
CANDIDATE_FORCE = re.compile(r'(force|f\b)', re.I)

def pick_column(df, pattern):
    for c in df.columns:
        if pattern.search(str(c)): return c
    return None

def uniformize_time(t, y):
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError("Non-increasing time vector.")
    t_uni = np.arange(t[0], t[-1] + 1e-12, dt)
    y_uni = np.interp(t_uni, t, y)
    return t_uni, y_uni, dt

def sgv_derivative(y, dt, order=1, window=SG_WINDOW, poly=SG_POLY):
    window = max(window, poly + 2 + (window % 2 == 0))  # ensure odd and >= poly+2
    return savgol_filter(y, window_length=window, polyorder=poly,
                         deriv=order, delta=dt, mode='interp')

def dominant_frequency(u, dt, fmin=FMIN_PEAK, fmax=None):
    n = len(u)
    U = rfft(u - np.mean(u))
    freqs = rfftfreq(n, dt)
    if fmax is None: fmax = 0.5/dt
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask): return None
    idx = np.argmax(np.abs(U[mask]))
    return freqs[mask][idx]

def physics_channels(df, file_hint=""):
    # Columns
    c_t = pick_column(df, CANDIDATE_TIME)
    c_u = pick_column(df, CANDIDATE_DISP)
    c_v = pick_column(df, CANDIDATE_VEL)
    c_a = pick_column(df, CANDIDATE_ACC)
    c_f = pick_column(df, CANDIDATE_FORCE)
    if c_t is None:
        raise ValueError(f"[{file_hint}] no time column found.")
    if c_u is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2: c_u = numeric_cols[1]
        else: raise ValueError(f"[{file_hint}] no displacement-like column found.")
    # Raw vectors
    t_raw = df[c_t].to_numpy(float)
    u_raw = df[c_u].to_numpy(float)
    t, u, dt = uniformize_time(t_raw, u_raw)
    # Velocity and acceleration
    if c_v is not None and pd.api.types.is_numeric_dtype(df[c_v]):
        v = np.interp(t, t_raw, df[c_v].to_numpy(float))
    else:
        v = sgv_derivative(u, dt, order=1)
    if c_a is not None and pd.api.types.is_numeric_dtype(df[c_a]):
        a = np.interp(t, t_raw, df[c_a].to_numpy(float))
    else:
        a = sgv_derivative(u, dt, order=2)
    # Force (free vibration -> zero if absent)
    if c_f is not None and pd.api.types.is_numeric_dtype(df[c_f]):
        f = np.interp(t, t_raw, df[c_f].to_numpy(float))
    else:
        f = np.zeros_like(t)
    # Mass / stiffness
    m = 1.0 if MASS is None else MASS
    if STIFFNESS is not None:
        k = STIFFNESS
    else:
        fpk = dominant_frequency(u, dt, fmin=FMIN_PEAK)
        if fpk is None:  # fallback ~1 Hz
            k = m*(2*np.pi*1.0)**2
        else:
            k = m*(2*np.pi*fpk)**2
    H_hat = f - m*a - k*u
    out = pd.DataFrame({
        "time": t, "f": f, "u": u, "v": v, "a": a, "H_hat": H_hat
    })
    meta = {"dt": dt, "mass_used": m, "stiffness_used": k}
    return out, meta

def make_windows(df, W, H_for, stride):
    Xs, Ys = [], []
    data = df[["f","u","v","a","H_hat"]].to_numpy()
    u = df["u"].to_numpy()
    N = len(df)
    for start in range(0, N - (W + H_for) + 1, stride):
        end = start + W
        Xs.append(data[start:end, :])
        Ys.append(u[end:end+H_for])
    return np.array(Xs), np.array(Ys)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    excel_files = []
    for ext in ("*.xlsx", "*.xls"):
        excel_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    if not excel_files:
        print("No Excel files found.")
        return

    summary = []
    for path in sorted(excel_files):
        try:
            xls = pd.ExcelFile(path)
            best_df, best_score = None, -1
            for sh in xls.sheet_names:
                df = pd.read_excel(path, sheet_name=sh)
                num_cols = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)
                if num_cols > best_score and len(df) > 5:
                    best_df, best_score = df, num_cols
            if best_df is None:
                print(f"Skip (no usable sheet): {os.path.basename(path)}"); continue

            out_df, meta = physics_channels(best_df, os.path.basename(path))
            base = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
            csv_path = os.path.join(OUTPUT_DIR, f"{base}__phys.csv")
            out_df.to_csv(csv_path, index=False)

            if MAKE_WINDOWS:
                X, Y = make_windows(out_df, W, H_FOR, STRIDE)
                np.savez_compressed(os.path.join(
                    OUTPUT_DIR, f"{base}__windows_W{W}_H{H_FOR}.npz"), X=X, Y=Y)

            print(f"Processed: {os.path.basename(path)} "
                  f"(m={meta['mass_used']:.4g}, k={meta['stiffness_used']:.4g}, dt={meta['dt']:.4g}s)")

            summary.append({
                "file": os.path.basename(path),
                "n_samples": len(out_df),
                **meta
            })
        except Exception as e:
            print(f"[ERROR] {os.path.basename(path)}: {e}")

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "summary_preprocessing.csv"), index=False)
        print(f"\nWrote summary to {os.path.join(OUTPUT_DIR, 'summary_preprocessing.csv')}")

if __name__ == "__main__":
    main()

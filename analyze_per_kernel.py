#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_per_kernel.py
---------------------
Reads arrays exported by ml_solver_v2.py and produces:
  • per-kernel CSV with residual (raw/normalized/acc), forecast, regression, and classification stats
  • bar plots for residuals (raw / normalized / acc), forecast RMSE, regression RMSE, and per-kernel recall

Inputs (expected in ./results/ml/arrays):
  - y_true_forecast.npy          [N, H]
  - y_pred_forecast.npy          [N, H]
  - a_true.npy                   [N, M]
  - a_pred.npy                   [N, M]
  - cls_true.npy                 [N]
  - cls_pred.npy                 [N]
  - residuals.npy                [N, W2]                 (raw residuals used in plots)
  - residuals_normalized.npy     [N, W2]   (optional)
  - residuals_acc.npy            [N, W2]   (optional)

Also reads:
  - ./results/ml/confusion_matrix.csv   (to fetch kernel names for labeling)

Outputs (under ./results/ml/analysis):
  - per_kernel_metrics.csv
  - bars_residual_std_raw.png
  - bars_residual_std_norm.png          (if residuals_normalized.npy exists)
  - bars_residual_std_acc.png           (if residuals_acc.npy exists)
  - bars_forecast_rmse.png
  - bars_regression_rmse.png
  - bars_per_kernel_recall.png
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("./results/ml_v8")
ARR  = ROOT / "arrays"
OUTD = ROOT / "analysis"

def _safe_load(path: Path):
    if path.exists():
        return np.load(path)
    return None

def _bar_plot(values, labels, ylabel, title, out_png, yerr=None):
    plt.figure(figsize=(10, 4))
    x = np.arange(len(labels))
    if yerr is not None:
        plt.bar(x, values, yerr=yerr, capsize=4)
    else:
        plt.bar(x, values)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def main():
    OUTD.mkdir(parents=True, exist_ok=True)

    # ---- Load required arrays
    y_true_fore = np.load(ARR / "y_true_forecast.npy")   # [N, H]
    y_pred_fore = np.load(ARR / "y_pred_forecast.npy")   # [N, H]
    a_true      = np.load(ARR / "a_true.npy")            # [N, M]
    a_pred      = np.load(ARR / "a_pred.npy")            # [N, M]
    cls_true    = np.load(ARR / "cls_true.npy")          # [N]
    cls_pred    = np.load(ARR / "cls_pred.npy")          # [N]
    residuals   = np.load(ARR / "residuals.npy")         # [N, W2]

    residuals_norm = _safe_load(ARR / "residuals_normalized.npy")  # optional
    residuals_acc  = _safe_load(ARR / "residuals_acc.npy")         # optional

    # ---- Kernel names (labels)
    cm_df = pd.read_csv(ROOT / "confusion_matrix.csv", index_col=0)
    kernel_names = cm_df.index.tolist()
    K = len(kernel_names)

    # ---- Pre-allocate metrics containers
    rows = []
    # For plots
    bars_res_std_raw  = []
    bars_res_std_norm = []
    bars_res_std_acc  = []
    bars_fore_rmse    = []
    bars_reg_rmse     = []
    bars_recall       = []

    for kid, kname in enumerate(kernel_names):
        idx = np.where(cls_true == kid)[0]
        if len(idx) == 0:
            # still append NaNs so bar charts stay aligned
            rows.append({
                "kernel": kname, "n_samples": 0,
                "cls_recall": np.nan, "cls_precision": np.nan,
                "forecast_MAE": np.nan, "forecast_RMSE": np.nan,
                "regression_MAE": np.nan, "regression_RMSE": np.nan,
                "res_mean_raw": np.nan, "res_std_raw": np.nan,
                "res_mean_norm": np.nan, "res_std_norm": np.nan,
                "res_mean_acc": np.nan, "res_std_acc": np.nan,
            })
            bars_res_std_raw.append(np.nan)
            bars_fore_rmse.append(np.nan)
            bars_reg_rmse.append(np.nan)
            bars_recall.append(np.nan)
            if residuals_norm is not None:
                bars_res_std_norm.append(np.nan)
            if residuals_acc is not None:
                bars_res_std_acc.append(np.nan)
            continue

        # classification: recall (true positives over actual) & precision (true positives over predicted)
        tp   = np.sum((cls_true[idx] == kid) & (cls_pred[idx] == kid))
        fn   = np.sum((cls_true[idx] == kid) & (cls_pred[idx] != kid))
        # predicted as this kernel:
        idx_pred_k = np.where(cls_pred == kid)[0]
        fp   = np.sum((cls_true[idx_pred_k] != kid) & (cls_pred[idx_pred_k] == kid))
        recall    = tp / (tp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)

        # forecast errors for this kernel
        yf_k = y_true_fore[idx]   # [Nk, H]
        yp_k = y_pred_fore[idx]   # [Nk, H]
        fore_mae  = float(np.mean(np.abs(yp_k - yf_k)))
        fore_rmse = float(np.sqrt(np.mean((yp_k - yf_k)**2)))

        # regression errors for this kernel (amplitudes)
        at_k = a_true[idx]        # [Nk, M]
        ap_k = a_pred[idx]        # [Nk, M]
        reg_mae  = float(np.mean(np.abs(ap_k - at_k)))
        reg_rmse = float(np.sqrt(np.mean((ap_k - at_k)**2)))

        # residuals (raw)
        rr_k = residuals[idx].ravel()
        res_mean_raw = float(np.mean(rr_k))
        res_std_raw  = float(np.std(rr_k))

        # optional: normalized residuals
        res_mean_norm = res_std_norm = np.nan
        if residuals_norm is not None:
            rn_k = residuals_norm[idx].ravel()
            res_mean_norm = float(np.mean(rn_k))
            res_std_norm  = float(np.std(rn_k))

        # optional: acceleration-based residuals
        res_mean_acc = res_std_acc = np.nan
        if residuals_acc is not None:
            ra_k = residuals_acc[idx].ravel()
            res_mean_acc = float(np.mean(ra_k))
            res_std_acc  = float(np.std(ra_k))

        # accumulate CSV row
        rows.append({
            "kernel": kname,
            "n_samples": int(len(idx)),
            "cls_recall": float(recall),
            "cls_precision": float(precision),
            "forecast_MAE": float(fore_mae),
            "forecast_RMSE": float(fore_rmse),
            "regression_MAE": float(reg_mae),
            "regression_RMSE": float(reg_rmse),
            "res_mean_raw": res_mean_raw,
            "res_std_raw": res_std_raw,
            "res_mean_norm": res_mean_norm,
            "res_std_norm": res_std_norm,
            "res_mean_acc": res_mean_acc,
            "res_std_acc": res_std_acc,
        })

        # bars data
        bars_res_std_raw.append(res_std_raw)
        bars_fore_rmse.append(fore_rmse)
        bars_reg_rmse.append(reg_rmse)
        bars_recall.append(recall)
        if residuals_norm is not None:
            bars_res_std_norm.append(res_std_norm)
        if residuals_acc is not None:
            bars_res_std_acc.append(res_std_acc)

    # ---- Write CSV
    out_csv = OUTD / "per_kernel_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote per-kernel metrics to {out_csv}")

    # ---- Plots
    # residual std (raw)
    _bar_plot(
        values=bars_res_std_raw, labels=kernel_names,
        ylabel="Std (raw)", title="Physics Residual Std per Kernel (raw)",
        out_png=OUTD / "bars_residual_std_raw.png"
    )
    # residual std (normalized) if available
    if residuals_norm is not None:
        _bar_plot(
            values=bars_res_std_norm, labels=kernel_names,
            ylabel="Std (normalized)", title="Physics Residual Std per Kernel (normalized)",
            out_png=OUTD / "bars_residual_std_norm.png"
        )
    # residual std (acc-based) if available
    if residuals_acc is not None:
        _bar_plot(
            values=bars_res_std_acc, labels=kernel_names,
            ylabel="Std (acc-based)", title="Physics Residual Std per Kernel (acceleration-based)",
            out_png=OUTD / "bars_residual_std_acc.png"
        )
    # forecast RMSE per kernel
    _bar_plot(
        values=bars_fore_rmse, labels=kernel_names,
        ylabel="RMSE", title="Forecast RMSE per Kernel",
        out_png=OUTD / "bars_forecast_rmse.png"
    )
    # regression RMSE per kernel
    _bar_plot(
        values=bars_reg_rmse, labels=kernel_names,
        ylabel="RMSE", title="Amplitude Regression RMSE per Kernel",
        out_png=OUTD / "bars_regression_rmse.png"
    )
    # per-kernel recall
    _bar_plot(
        values=bars_recall, labels=kernel_names,
        ylabel="Recall", title="Classification Recall per Kernel",
        out_png=OUTD / "bars_per_kernel_recall.png"
    )

if __name__ == "__main__":
    main()

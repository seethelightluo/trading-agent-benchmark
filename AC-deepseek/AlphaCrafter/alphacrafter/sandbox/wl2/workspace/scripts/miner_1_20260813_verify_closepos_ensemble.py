"""miner_1 2026-08-13: verify close_pos_20 artifact linkage + ensemble compatibility.
Check: (1) recoverable artifacts for the 8 active ensemble factors + close_pos_20;
(2) pairwise Spearman rho (rank-aligned, overlapping rows) of close_pos_20 vs each
active ensemble factor using REAL artifacts; report vs 0.5 gate."""
import json, glob, os
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

date_state = json.load(open("../persistent/date.json"))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
MIN_ASSETS = 8

ENSEMBLE = ["downbeta_spx_60", "mom20_volproxy60", "mom_20d_skip5", "gain_loss_20",
            "usdjpy_beta_cond_120x60", "volcluster_60", "range_pos_252", "calmness_20"]


def load_artifact(fid):
    """Return factor matrix (rows aligned to GRID) from .signal.npy or embedded daily_panel."""
    # 1) .signal.npy
    npy = f"factors/{fid}.signal.npy"
    if os.path.exists(npy):
        a = np.load(npy, allow_pickle=True)
        if isinstance(a, np.ndarray):
            if a.dtype == object and a.shape == ():
                a = a.item()
            if isinstance(a, dict):
                # embedded dict artifact with values
                if "values" in a:
                    return np.asarray(a["values"], dtype=float)
            else:
                return np.asarray(a, dtype=float)
    # 2) embedded daily_panel in JSON
    jp = f"factors/{fid}.json"
    if os.path.exists(jp):
        d = json.load(open(jp))
        m = d.get("validation", {}).get("metrics", {})
        if "daily_panel" in m:
            dp = m["daily_panel"]
            return np.asarray(dp.get("values", dp), dtype=float)
        if "signal" in m:
            return np.asarray(m["signal"], dtype=float)
    return None


def rank_align(mat, rows):
    a = mat[:rows]
    out = np.full_like(a, np.nan, dtype=float)
    for t in range(rows):
        row = a[t]
        valid = ~np.isnan(row)
        if valid.sum() < MIN_ASSETS:
            continue
        r = pd.Series(row[valid]).rank(pct=True).values
        out[t, valid] = r
    return out


def pair_rho(mat_a, mat_b):
    rows = min(mat_a.shape[0], mat_b.shape[0])
    a = rank_align(mat_a, rows)
    b = rank_align(mat_b, rows)
    rhos = []
    for t in range(rows):
        x, y = a[t], b[t]
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < MIN_ASSETS:
            continue
        c = pd.Series(x[ok]).corr(pd.Series(y[ok]))
        if np.isfinite(c):
            rhos.append(c)
    if not rhos:
        return None, 0
    return float(np.nanmean(rhos)), len(rhos)


print("=" * 78)
print("ARTIFACT RECOVERABILITY (2026-08-13 cycle, visible through", VISIBLE, ")")
print("=" * 78)
for fid in ENSEMBLE + ["close_pos_20"]:
    mat = load_artifact(fid)
    if mat is None:
        print(f"  {fid:28s} NO RECOVERABLE ARTIFACT")
    else:
        mat = np.asarray(mat, dtype=float)
        # align to GRID rows
        if mat.shape[0] > len(GRID):
            mat = mat[-len(GRID):]
        nan_last = np.isnan(mat[-1]).sum() if mat.shape[0] > 0 else 15
        print(f"  {fid:28s} shape={mat.shape} lastrow_nan={nan_last} lastrow_valid={15 - nan_last}")

print()
print("=" * 78)
print("close_pos_20 vs ACTIVE ENSEMBLE pairwise Spearman rho (rank-aligned, mean over dates)")
print("=" * 78)
cp = load_artifact("close_pos_20")
if cp is None:
    print("close_pos_20 artifact MISSING - cannot check")
else:
    cp = np.asarray(cp, dtype=float)
    if cp.shape[0] > len(GRID):
        cp = cp[-len(GRID):]
    for fid in ENSEMBLE:
        m = load_artifact(fid)
        if m is None:
            print(f"  vs {fid:28s} no artifact to compare")
            continue
        m = np.asarray(m, dtype=float)
        if m.shape[0] > len(GRID):
            m = m[-len(GRID):]
        rho, n = pair_rho(cp, m)
        flag = "  <-- OVER 0.5 GATE" if (rho is not None and abs(rho) >= 0.5) else ""
        print(f"  vs {fid:28s} rho={rho if rho is not None else float('nan'):+.4f}  n_dates={n}{flag}")

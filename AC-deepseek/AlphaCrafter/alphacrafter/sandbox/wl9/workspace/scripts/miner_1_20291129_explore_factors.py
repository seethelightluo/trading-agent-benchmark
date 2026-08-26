"""Miner1 2029-11-29: explore and validate new candidate factors on data up to visible end.

Universe: 15 tradable cross-asset instruments. Min 8 instruments per date for IC.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at horizon 10.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

VISIBLE_END = "2029-11-28"
STOCK_DIR = Path("../persistent/stock_data")
INDEX_DIR = Path("../persistent/index_data")
ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACROS = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
MIN_ASSETS = 8

def load_close(symbol, macro=False):
    d = INDEX_DIR if macro else STOCK_DIR
    df = pd.read_csv(Path(d) / f"{symbol}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE_END)]
    s = df.set_index("date")["close"].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s

def load_panel():
    return pd.concat({a: load_close(a) for a in ASSETS}, axis=1)

def load_macro():
    return {m: load_close(m, macro=True) for m in MACROS}

def fwd_ret_panel(panel, h):
    out = {}
    for a in panel.columns:
        s = panel[a].dropna()
        out[a] = s.shift(-h) / s - 1.0
    return pd.DataFrame(out).sort_index()

def ic_series(fvals, fwd):
    idx = fvals.index.intersection(fwd.index)
    ics = {}
    for t in idx:
        x, y = fvals.loc[t], fwd.loc[t]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            rho, _ = spearmanr(x[m], y[m])
            if np.isfinite(rho):
                ics[t] = rho
    return pd.Series(ics)

def rank_turnover(fvals, step=10):
    r = fvals.rank(axis=1)
    return float(r.diff(step).abs().mean().mean())

def compute_metrics(fvals, panel):
    fwd10 = fwd_ret_panel(panel, 10)
    ic = ic_series(fvals, fwd10)
    n = len(ic)
    if n < 20:
        return None
    mean_ic = float(ic.mean())
    std_ic = float(ic.std(ddof=1)) if n > 1 else np.nan
    icir = mean_ic / std_ic if std_ic and np.isfinite(std_ic) and std_ic > 0 else np.nan
    # expected direction sign chosen to maximize |IC|
    exp_dir = 1 if mean_ic >= 0 else -1
    hit = float((ic * exp_dir > 0).mean())
    cov = float(fvals.notna().sum().sum() / (fvals.shape[0] * fvals.shape[1]))
    turn = rank_turnover(fvals)
    decay = {}
    for h in [1, 2, 3, 5, 10, 20]:
        ih = ic_series(fvals, fwd_ret_panel(panel, h))
        decay[str(h)] = round(float(ih.mean()), 4) if len(ih) else None
    return {"n_ic_dates": int(n), "ic": round(mean_ic, 4), "icir": round(icir, 4),
            "ic_hit_ratio": round(hit, 4), "coverage": round(cov, 4),
            "turnover_10d_rank": round(turn, 3), "decay": decay,
            "passes": bool(abs(mean_ic) >= 0.0070 and abs(icir) >= 0.0840),
            "exp_dir": exp_dir}

def run(name, fvals, panel, results):
    m = compute_metrics(fvals, panel)
    if m is None:
        print(f"{name:28s} too few IC dates")
        return
    results[name] = m
    print(f"{name:28s} IC={m['ic']:+.4f} ICIR={m['icir']:+.3f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} cov={m['coverage']:.2f} turn={m['turnover_10d_rank']:.3f} "
          f"pass={m['passes']} dir={m['exp_dir']:+d}")
    print(f"   decay: {m['decay']}")

P = load_panel()
MAC = load_macro()
print(f"Panel: {P.shape[0]} dates x {P.shape[1]} assets, {P.index.min().date()} -> {P.index.max().date()}")
print(f"Full history assets: {(P.notna().sum()>=1800).sum()}/{P.shape[1]}")

results = {}

# Candidate 1: RV_10_60 - return/vol quality: 10d momentum normalized by 60d realized vol.
print("\n=== RV_10_60: 10d mom / 60d realized vol ===")
rv = pd.DataFrame(index=P.index, columns=P.columns, dtype=float)
for a in P.columns:
    s = P[
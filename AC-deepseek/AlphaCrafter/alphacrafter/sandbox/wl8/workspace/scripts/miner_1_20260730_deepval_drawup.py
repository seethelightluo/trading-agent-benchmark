"""miner_1 deep validation: drawup_40 / drawup_60 / drawup_120 / dist_low_60.
Checks regime stability, mutual correlations, and library rho. Decides persistence set."""
import sys, time, json, base64, zlib, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    load_closes, load_index, factor_panel, fwd_returns, ic_series,
    coverage, turnover_rank, IC_GATE, ICIR_GATE,
)

t0 = time.time()
close, vol, open_, high, low = load_closes()
macro = {
    "DXY": load_index("DXY"), "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"), "EURUSD": load_index("EURUSD"),
    "VIX": load_index("VIX"),
}


def f_drawup(c, v, o, h, l, m, win=60):
    roll = c.rolling(win)
    maxup = (c / roll.max() - 1.0).abs()
    maxdn = (c / roll.min() - 1.0).abs()
    denom = (maxup + maxdn).replace(0, np.nan)
    return (maxup / denom)


def f_dist_low(c, v, o, h, l, m, win=60):
    return (c / c.rolling(win).min() - 1.0)


FACTORS = {
    "drawup_40": {"fn": f_drawup, "params": {"win": 40}},
    "drawup_60": {"fn": f_drawup, "params": {"win": 60}},
    "drawup_120": {"fn": f_drawup, "params": {"win": 120}},
    "dist_low_60": {"fn": f_dist_low, "params": {"win": 60}},
}

d = json.load(open("factors/usdcny_beta_60.json"))
raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
lib_panel = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
lib_panel.index = pd.DatetimeIndex(lib_panel.index)

fwd10 = fwd_returns(close, 10)
panels = {}
for fid, spec in FACTORS.items():
    panels[fid] = factor_panel(spec["fn"], close, vol, open_, high, low, macro, **spec["params"])

REGIMES = {
    "2020-2021 COVID/recovery": ("2020-01-01", "2021-12-31"),
    "2022-2023 tightening/AI": ("2022-01-01", "2023-12-31"),
    "2024-2026-07 crypto/commodity": ("2024-01-01", "2026-07-30"),
    "2025-07-2026-07 last_12m": ("2025-07-30", "2026-07-30"),
}


def spearman_pooled(a_panel, b_panel):
    common = a_panel.index.intersection(b_panel.index)
    cols = [c for c in a_panel.columns if c in b_panel.columns]
    a = a_panel.loc[common, cols].values.ravel()
    b = b_panel.loc[common, cols].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 200:
        return np.nan, int(m.sum())
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())), int(m.sum())


for fid, panel in panels.items():
    icm = ic_series(panel, fwd10)
    ic = float(icm.mean()); icir = float(icm.mean() / icm.std())
    hit = float((icm < 0).mean()) if ic < 0 else float((icm > 0).mean())
    print(f"\n=== {fid}: full IC={ic:+.4f} ICIR={icir:+.4f} hit={hit:.3f} n={len(icm)} ===", flush=True)
    for rname, (a, b) in REGIMES.items():
        sub = icm.loc[a:b]
        if len(sub) > 5:
            print(f"  {rname}: IC={sub.mean():+.4f} ICIR={sub.mean()/sub.std():+.3f} n={len(sub)}", flush=True)
    rho, n = spearman_pooled(panel, lib_panel)
    print(f"  rho vs usdcny_beta_60: {rho:.4f} (n={n})", flush=True)

print("\n--- mutual pooled spearman (candidate vs candidate) ---", flush=True)
fids = list(panels.keys())
for i in range(len(fids)):
    for j in range(i + 1, len(fids)):
        r, n = spearman_pooled(panels[fids[i]], panels[fids[j]])
        print(f"  {fids[i]} vs {fids[j]}: {r:.4f} (n={n})", flush=True)

print(f"\ndone in {time.time()-t0:.1f}s", flush=True)

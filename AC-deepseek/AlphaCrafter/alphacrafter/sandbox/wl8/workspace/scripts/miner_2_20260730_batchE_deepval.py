"""miner_2 batch-E deep-validation of passers (2026-07-30).
Passers: abn_vol_20, abn_vol_z_20, up_ratio_20, body_10, dd_vol_20.
Checks: per-asset coverage, regime IC, pairwise pooled spearman among passers
and vs usdcny_beta_60 library.
"""
import sys, time, json, base64, zlib, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (
    load_closes, load_index, factor_panel, fwd_returns, ic_series,
    coverage, turnover_rank, IC_GATE, ICIR_GATE,
)

close, vol, open_, high, low = load_closes()
macro = {
    "DXY": load_index("DXY"), "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"), "EURUSD": load_index("EURUSD"),
    "VIX": load_index("VIX"),
}


def f_abn_vol(c, v, o, h, l, m, win=20):
    return (v / v.rolling(win).mean()).replace([np.inf, -np.inf], np.nan)


def f_abn_vol_z(c, v, o, h, l, m, win=20):
    return ((v - v.rolling(win).mean()) / v.rolling(win).std()).replace([np.inf, -np.inf], np.nan)


def f_up_ratio(c, v, o, h, l, m, win=20):
    return (c.pct_change() > 0).rolling(win).mean()


def f_body_10(c, v, o, h, l, m, win=10):
    rng = (h - l).replace(0, np.nan)
    return ((c - o).abs() / rng).rolling(win).mean()


def f_dd_vol_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    roll_max = c.rolling(win).max()
    dd = (c / roll_max - 1.0).rolling(win).min()
    return (dd / r.rolling(win).std()).replace([np.inf, -np.inf], np.nan)


FACTORS = {
    "abn_vol_20": {"fn": f_abn_vol, "params": {"win": 20}},
    "abn_vol_z_20": {"fn": f_abn_vol_z, "params": {"win": 20}},
    "up_ratio_20": {"fn": f_up_ratio, "params": {"win": 20}},
    "body_10": {"fn": f_body_10, "params": {"win": 10}},
    "dd_vol_20": {"fn": f_dd_vol_20, "params": {"win": 20}},
}


def load_lib():
    lib = {}
    d = json.load(open("factors/usdcny_beta_60.json"))
    art = d["validation"]["signal_artifact"]
    raw = base64.b64decode(art["data"])
    p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index)
    lib["usdcny_beta_60"] = p
    return lib


lib = load_lib()


def spearman_pooled(a_panel, b_panel, min_n=50):
    common = a_panel.index.intersection(b_panel.index)
    cols = [c for c in a_panel.columns if c in b_panel.columns]
    a = a_panel.loc[common, cols].values.ravel()
    b = b_panel.loc[common, cols].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < min_n:
        return np.nan, int(m.sum())
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())), int(m.sum())


fwd10 = fwd_returns(close, 10)
panels = {}
for fid, spec in FACTORS.items():
    panels[fid] = factor_panel(spec["fn"], close, vol, open_, high, low, macro, **spec["params"])

# per-asset coverage of volume-based factors
print("=== per-asset non-NaN fraction (abn_vol_20 / up_ratio_20 / body_10 / dd_vol_20) ===", flush=True)
for a in panels["abn_vol_20"].columns:
    print(f"  {a:10s} abnvol={panels['abn_vol_20'][a].notna().mean():.2f} up={panels['up_ratio_20'][a].notna().mean():.2f} "
          f"body={panels['body_10'][a].notna().mean():.2f} dd={panels['dd_vol_20'][a].notna().mean():.2f}", flush=True)

# vol==0 check
print("\n=== volume==0 counts ===", flush=True)
for a in vol.columns:
    z = (vol[a] == 0).sum()
    if z > 0:
        print(f"  {a}: zero-vol days={z}", flush=True)

# regime IC at h=10
REGIMES = [("2020-2021", "2020-01-01", "2021-12-31"),
           ("2022-2023", "2022-01-01", "2023-12-31"),
           ("2024-2026", "2024-01-01", "2026-07-30")]
print("\n=== regime IC (h=10) ===", flush=True)
for fid, panel in panels.items():
    icm = ic_series(panel, fwd10)
    row = []
    for name, d0, d1 in REGIMES:
        sub = icm.loc[(icm.index >= d0) & (icm.index <= d1)]
        row.append(f"{name}: ic={sub.mean():+.4f} icir={(sub.mean()/sub.std() if sub.std()>0 else np.nan):+.3f} n={len(sub)}")
    print(f"  {fid:12s} " + " | ".join(row), flush=True)

# pairwise spearman among passers and vs library
print("\n=== pairwise pooled spearman ===", flush=True)
fids = list(panels.keys())
for i, fi in enumerate(fids):
    for fj in fids[i + 1:]:
        r, n = spearman_pooled(panels[fi], panels[fj])
        print(f"  {fi} vs {fj}: rho={r:.4f} (n={n})", flush=True)
for fid in fids:
    r, n = spearman_pooled(panels[fid], lib["usdcny_beta_60"])
    print(f"  {fid} vs usdcny_beta_60: rho={r:.4f} (n={n})", flush=True)
print("done", flush=True)

"""miner_1 cycle6 -- test daily-abs-rho hypothesis vs gate eviction numbers."""
import json, base64, zlib, io
import numpy as np
import pandas as pd

def load_panel(fid, base="factors"):
    d = json.load(open(f"{base}/{fid}.json"))
    raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
    p = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
    p.index = pd.DatetimeIndex(p.index)
    return p

def daily_rho_stats(a, b, min_assets=5):
    common = a.index.intersection(b.index)
    cols = [c for c in a.columns if c in b.columns]
    rhos = []
    for dt in common:
        x = a.loc[dt, cols]; y = b.loc[dt, cols]
        m = x.notna() & y.notna()
        if m.sum() >= min_assets:
            v = x[m].rank().corr(y[m].rank())
            if np.isfinite(v):
                rhos.append(v)
    rhos = np.array(rhos)
    if len(rhos) < 10:
        return dict(n=len(rhos))
    return dict(n=len(rhos), mean=float(rhos.mean()), mean_abs=float(np.abs(rhos).mean()),
                max_abs=float(np.abs(rhos).max()), p50_abs=float(np.median(np.abs(rhos))))

yb = load_panel("yield_beta_cond_60x20")
print(f"{'candidate':24s} {'n_days':>6s} {'mean':>7s} {'mean|rho|':>9s} {'median|rho|':>11s} {'max|rho|':>8s}")
for cand, base in [("eff_ratio_20","factors/evicted"),("down_vol_ratio_20x60","factors/evicted"),
                   ("ret_kurt_30","factors/evicted"),("hl_pos_150","factors"),("hl_pos_180","factors"),
                   ("mom_10d_skip5","factors"),("vix_beta_cond_60x20","factors")]:
    p = load_panel(cand, base)
    s = daily_rho_stats(p, yb)
    if "mean" in s:
        print(f"{cand:24s} {s['n']:6d} {s['mean']:7.4f} {s['mean_abs']:9.4f} {s['p50_abs']:11.4f} {s['max_abs']:8.4f}")
    else:
        print(f"{cand:24s} n={s['n']}")

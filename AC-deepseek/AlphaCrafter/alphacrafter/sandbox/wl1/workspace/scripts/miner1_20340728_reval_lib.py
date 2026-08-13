"""miner_1 2034-07-28: continuous re-validation of all EFFECTIVE library factors.
Recomputes each factor signal from real OHLC/VIX data (calculation expressions read
from the persisted JSON), then computes daily cross-sectional Spearman IC vs forward
returns on the 15-asset panel through 2034-07-28.

Gates (shared benchmark-wide): |IC1| >= 0.0070 and |ICIR1| >= 0.0840.
Reports full-window, recent-1y and recent-120d metrics to assess drift.
"""
import json
import numpy as np
import pandas as pd
import glob

panel = pd.read_pickle("scripts/panel_cache_20340728.pkl")
close = panel["close"]
open_ = panel["open"]
high = panel["high"]
low = panel["low"]
ret = panel["ret"]
macro = panel["macro"]
vix = macro["VIX"].reindex(close.index).ffill()

TRADABLE = list(close.columns)
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def ic_series(signal, fwd):
    """Daily cross-sectional Spearman IC (>=8 valid instruments per date)."""
    sig_r = signal.rank(axis=1)
    dates, ics = [], []
    for dt in signal.index:
        s = sig_r.loc[dt]
        f = fwd.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() >= 8:
            ic = s[m].corr(f[m], method="spearman")
            if np.isfinite(ic):
                ics.append(ic)
                dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))


def stats(ics):
    if len(ics) == 0:
        return dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
    sd = float(ics.std(ddof=1))
    return dict(ic=float(ics.mean()),
                icir=float(ics.mean() / sd) if sd > 0 else np.nan,
                hit=float((ics > 0).mean()), n=len(ics))


def compute_library_signal(f):
    expr = f["calculation"]["expression"]
    p = f["calculation"].get("parameters", {})
    if f["factor_id"].startswith("miner2_20260715_nclv"):
        nd = p.get("nd", 1)
        rng = high.rolling(nd).max() - low.rolling(nd).min()
        return -(close - low.rolling(nd).min()) / rng.replace(0, np.nan)
    if f["factor_id"].startswith("miner2_20260715_rev") and "vs" not in f["factor_id"]:
        nd = p.get("nd", 1)
        return -(np.log(close) - np.log(close.shift(nd)))
    if f["factor_id"] == "miner2_20260715_rev_1d_vs":
        nd = p.get("nd", 1); vw = p.get("vol_window", 20)
        r = np.log(close).diff(nd)
        return -r / ret.rolling(vw).std().replace(0, np.nan)
    if f["factor_id"] == "miner2_20260715_id_rev_1d":
        return -(close / open_ - 1.0)
    if f["factor_id"] == "miner2_20260715_nbody_1d":
        return -(close - open_) / (high - low).replace(0, np.nan)
    if f["factor_id"] == "mom_120d_skip5":
        return close.shift(5) / close.shift(125) - 1.0
    if f["factor_id"] == "vol_of_vol20x60":
        return ret.rolling(20).std().rolling(60).std()
    if f["factor_id"] == "vix_beta_cond_60x20":
        vix_ret = vix.pct_change()
        beta = ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var().replace(0, np.nan)
        return -beta * (vix / vix.shift(20) - 1.0)
    raise ValueError(f"unknown factor {f['factor_id']}")


def horizon_ics(signal, hz):
    fwd = close.shift(-hz) / close - 1.0
    return ic_series(signal, fwd)


def window_stats(ics_all, sl):
    return stats(ics_all.loc[sl])


results = {}
for fp in sorted(glob.glob("factors/*.json")):
    if ".bak" in fp:
        continue
    f = json.load(open(fp))
    if f.get("validation", {}).get("status") != "EFFECTIVE":
        continue
    fid = f["factor_id"]
    try:
        sig = compute_library_signal(f).reindex(close.index)
    except Exception as e:
        print(fid, "COMPUTE ERROR", e)
        continue
    h1 = horizon_ics(sig, 1)
    full = stats(h1)
    rec1y = stats(h1.loc[h1.index[-260:]])
    rec120 = stats(h1.loc[h1.index[-120:]])
    # also h5/h10 for decay context
    h5 = stats(horizon_ics(sig, 5))
    h10 = stats(horizon_ics(sig, 10))
    passed = (abs(full["ic"]) >= GATE_IC) and (abs(full["icir"]) >= GATE_ICIR)
    rec_passed = (abs(rec120["ic"]) >= GATE_IC) and (abs(rec120["icir"]) >= GATE_ICIR)
    results[fid] = dict(full=full, rec1y=rec1y, rec120=rec120, h5=h5, h10=h10,
                        passed=passed, rec_passed=rec_passed,
                        cov=float(sig.notna().mean(axis=1).mean()))
    print(f"{fid:32s} full IC={full['ic']:+.4f} ICIR={full['icir']:+.3f} hit={full['hit']:.3f} n={full['n']:4d} | "
          f"rec1y IC={rec1y['ic']:+.4f} ICIR={rec1y['icir']:+.3f} | "
          f"rec120 IC={rec120['ic']:+.4f} ICIR={rec120['icir']:+.3f} | "
          f"h5 IC={h5['ic']:+.4f} h10 IC={h10['ic']:+.4f} | cov={results[fid]['cov']:.2f} | "
          f"{'PASS' if passed else 'FAIL'} {'REC-PASS' if rec_passed else 'REC-FAIL'}")

json.dump(results, open("scripts/miner1_20340728_reval_lib.json", "w"), indent=1, default=str)
print("\nSaved scripts/miner1_20340728_reval_lib.json")

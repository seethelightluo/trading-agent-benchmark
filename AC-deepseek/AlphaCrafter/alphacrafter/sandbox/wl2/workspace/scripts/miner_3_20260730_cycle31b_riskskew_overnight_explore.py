"""miner_3 2026-07-30 cycle 31b: return-structure & risk-asymmetry family (fixed per-asset).

Fixes: all rolling quantities computed per-asset on the asset's OWN calendar via
per_asset() so union-index NaN gaps never poison rolling windows (the previous run
computed skew/downside-ratio/overnight splits on the union panel -> coverage collapse).
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, per_asset, forward_returns, compute_ic,
                         validate_factor, report, VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

lib = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    lib[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)
print(f"library loaded: {list(lib.keys())}; panel {panel.shape}")


def load_ohlc(field):
    out = {}
    for a in panel.columns:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv", parse_dates=["date"])
        df = df[df["date"] <= pd.Timestamp(VISIBLE_THROUGH)].sort_values("date")
        out[a] = pd.Series(df[field].astype(float).values,
                           index=pd.to_datetime(df["date"]), name=a)
    return pd.DataFrame(out, index=panel.index).sort_index()


open_p = load_ohlc("open")
high_p = load_ohlc("high")
low_p = load_ohlc("low")

# ---- per-asset own-calendar return helpers ---------------------------------
def _r(s):
    return s.pct_change()


def _down_dev(s, w=60, minp=40):
    r = s.pct_change()
    return r.where(r < 0, 0.0).pow(2).rolling(w, min_periods=minp).mean().pow(0.5)


def _vol(s, w=60, minp=40):
    return s.pct_change().rolling(w, min_periods=minp).std()


def _down_ratio(s, w=60, minp=40):
    r = s.pct_change()
    dd = r.where(r < 0, 0.0).pow(2).rolling(w, min_periods=minp).mean().pow(0.5)
    v = r.rolling(w, min_periods=minp).std()
    return dd / v


def _skew(s, w=60, minp=40):
    return s.pct_change().rolling(w, min_periods=minp).skew()


def _maxdd(s, w=60, minp=40):
    r = s.pct_change()
    dd = np.empty(len(r)) * np.nan
    xa = r.values
    for i in range(len(xa)):
        if i < minp - 1:
            continue
        seg = xa[max(0, i - w + 1): i + 1]
        if np.isnan(seg).any():
            continue
        peak = np.maximum.accumulate(seg)[-1]
        dd[i] = seg[-1] / peak - 1.0
    return pd.Series(dd, index=r.index).abs()


def _reward_dd(s, w=60, minp=40):
    dd = _maxdd(s, w, minp)
    ret = s / s.shift(w) - 1.0
    return ret / dd.replace(0, np.nan)


def _ovn_cum(s, op, w=20, minp=12):
    ovn = op / s.shift(1) - 1.0
    return (1 + ovn).rolling(w, min_periods=minp).apply(np.prod, raw=True) - 1.0


def _int_cum(s, op, w=20, minp=12):
    intr = s / op - 1.0
    return (1 + intr).rolling(w, min_periods=minp).apply(np.prod, raw=True) - 1.0


def _ovn_share(s, op, w=20, minp=12):
    oc = _ovn_cum(s, op, w, minp)
    ic = _int_cum(s, op, w, minp)
    return oc.abs() / (oc.abs() + ic.abs())


def _gap_fade(s, op, w=20, minp=12):
    gap = op / s.shift(1) - 1.0
    intr = s / op - 1.0
    return gap.rolling(w, min_periods=minp).corr(intr)


f_down_dev = per_asset(panel, _down_dev)
f_vol60 = per_asset(panel, _vol)
f_down_ratio = per_asset(panel, _down_ratio)
f_skew = per_asset(panel, _skew)
f_maxdd = per_asset(panel, _maxdd)
f_reward_dd = per_asset(panel, _reward_dd)
f_ovn_cum = per_asset(panel, _ovn_cum, open_p)
f_int_cum = per_asset(panel, _int_cum, open_p)
f_ovn_share = per_asset(panel, _ovn_share, open_p)
f_gap_fade = per_asset(panel, _gap_fade, open_p)

cands = {
    "downside_dev_60": f_down_dev,
    "downside_ratio_60": f_down_ratio,
    "skew_60": f_skew,
    "maxdd_60": f_maxdd,
    "reward_dd_60": f_reward_dd,
    "overnight_cum_20": f_ovn_cum,
    "intraday_cum_20": f_int_cum,
    "overnight_share_20": f_ovn_share,
    "gap_fade_20": f_gap_fade,
}

print("\n=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=lib, fwd_cache=fwd_cache)
    p = report(name, m)
    results[name] = {"metrics": m, "pass": p}
    print(f"    decay: {m['decay_ic_by_horizon']}")
    print(f"    pairwise: {m.get('library_pairwise_corr')}")

print("\n=== REGIME BREAKDOWN (10d IC by sub-period) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd_cache[str(ADM_H)]).dropna()
    parts = []
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = ic_ser[(ic_ser.index >= r0) & (ic_ser.index <= r1)]
        if len(sub) >= 30:
            sd = sub.std()
            parts.append(f"{r0[:4]}:ic={sub.mean():+.4f}/icir={(sub.mean()/sd if sd>0 else 0):+.3f}/n={len(sub)}")
    print(f"  {name:22s} | " + " | ".join(parts))

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/_miner3_cycle31b_explore_results.json", "w"), indent=1, default=float)
print("\nDONE cycle31b")

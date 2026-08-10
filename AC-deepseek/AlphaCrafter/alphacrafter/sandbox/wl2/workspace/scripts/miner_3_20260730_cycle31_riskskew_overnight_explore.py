"""miner_3 2026-07-30 cycle 31: explore return-structure & risk-asymmetry factor family.

Motivation: the ACTIVE library holds vol-damped momentum (mom20_volproxy60), DXY-beta
conditioning (dxy_beta_cond_60x20) and trend calmness (calmness_20). Raw momentum,
carry, range-position, efficiency-ratio, vol-surge and intraday-drift families were
already explored/evicted/quarantined. This cycle targets a DIFFERENT information axis:
*where inside the return distribution / day structure the signal lives*:

  1) downside semi-deviation and downside/total-vol asymmetry (crash-risk pricing)
  2) rolling skewness (tail asymmetry)
  3) max drawdown and reward-per-drawdown (pain ratio)
  4) overnight (close->open) vs intraday (open->close) cumulative return split
     (behavioral anomaly: overnight risk premium vs intraday sentiment)
  5) gap-fade coherence: do gaps reverse or continue within the day? (rolling corr
     of overnight gap with same-day intraday return)

All computed per-asset on the asset's own calendar (no NaN gaps), reindexed to the
union panel. Visible cutoff 2026-07-29. Library = 3 ACTIVE persisted factors loaded
from real signal artifacts. Admission gate: |IC|>=0.007 AND |ICIR|>=0.084 at 10d.
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

# --- library = ACTIVE persisted factors (real artifacts) ---
lib = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20", "calmness_20"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    lib[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)
print(f"library loaded: {list(lib.keys())}; panel {panel.shape} "
      f"dates {panel.index.min().date()}..{panel.index.max().date()}")


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

# ===========================================================================
# Candidate constructions
# ===========================================================================
r = panel.pct_change()                      # per-asset own-calendar returns
vol_60 = r.rolling(60, min_periods=40).std()

# 1) downside semi-deviation 60d (RMS of negative returns)
down_ret = r.where(r < 0, 0.0)
f_down_dev = down_ret.pow(2).rolling(60, min_periods=40).mean().pow(0.5)

# 2) downside/total-vol asymmetry ratio 60d
f_down_ratio = f_down_dev / vol_60

# 3) rolling skewness 60d
f_skew = r.rolling(60, min_periods=40).skew()

# 4) max drawdown over 60d (positive magnitude)
def _maxdd_60(s, w=60, minp=40):
    roll = s.rolling(w, min_periods=minp)
    return roll.max() / roll.apply(lambda x: np.maximum.accumulate(x).max()) - 1.0


def _maxdd_panel(x, w=60, minp=40):
    dd = np.empty(len(x)) * np.nan
    xa = x.values
    for i in range(len(xa)):
        if i < minp - 1:
            continue
        seg = xa[max(0, i - w + 1): i + 1]
        if np.isnan(seg).any():
            continue
        peak = np.maximum.accumulate(seg)[-1]
        dd[i] = seg[-1] / peak - 1.0
    return pd.Series(dd, index=x.index)


f_maxdd = per_asset(panel, _maxdd_panel, 60, 40).abs()

# 5) reward-per-drawdown (pain ratio): 60d return / maxdd magnitude
ret_60 = panel / panel.shift(60) - 1.0
f_reward_dd = ret_60 / f_maxdd.replace(0, np.nan)

# 6/7) overnight & intraday cumulative returns over 20d
overnight = open_p / panel.shift(1) - 1.0      # close[t-1] -> open[t]
intraday = panel / open_p - 1.0                # open[t] -> close[t]
f_ovn_cum = (1 + overnight).rolling(20, min_periods=12).apply(np.prod, raw=True) - 1.0
f_int_cum = (1 + intraday).rolling(20, min_periods=12).apply(np.prod, raw=True) - 1.0

# 8) overnight share of total 20d |move|
f_ovn_share = f_ovn_cum.abs() / (f_ovn_cum.abs() + f_int_cum.abs())

# 9) gap-fade coherence 20d: corr(overnight gap, same-day intraday ret)
def _gap_fade_corr(w=20, minp=12):
    out = {}
    for a in panel.columns:
        c = panel[a].dropna()
        op = open_p[a].reindex(c.index).dropna()
        idx = c.index.intersection(op.index)
        cc, oo = c[idx], op[idx]
        gap = oo / cc.shift(1) - 1.0
        intr = cc / oo - 1.0
        corr = gap.rolling(w, min_periods=minp).corr(intr)
        out[a] = corr.reindex(panel.index)
    return pd.DataFrame(out, index=panel.index)


f_gap_fade = _gap_fade_corr()

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
          open("scripts/_miner3_cycle31_explore_results.json", "w"), indent=1, default=float)
print("\nDONE cycle31 exploration")

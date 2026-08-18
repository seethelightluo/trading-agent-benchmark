"""miner_3 2027-03-02 re-validation of currently effective factors.

Recomputes each ensemble factor on data through 2027-03-01 and checks the
admission gates: |IC| >= 0.007 and |ICIR| >= 0.084 at h=10 (15-asset universe).
Also reports decay profile, per-year IC, turnover, coverage.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, "scripts")
from miner3_lib_2027 import (load_panel, load_ohlc_volume, load_macro,
                             validate_factor, decay_profile, turnover_10d,
                             coverage, ADMISSION)

panel = load_panel()
ohlcv = load_ohlc_volume()
macro = load_macro()
print(f"panel dates: {panel.index.min().date()} .. {panel.index.max().date()}, "
      f"n_dates={len(panel)}, n_assets={panel.shape[1]}")
print(f"macro available: {sorted(macro.keys())}")


def ew_market_ret(panel, window=60):
    ret = panel.pct_change()
    ew = ret.mean(axis=1, skipna=True)
    return ew


def rolling_beta(asset_ret: pd.Series, mkt_ret: pd.Series, window: int) -> pd.Series:
    cov = asset_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    return cov / var.replace(0, np.nan)


# ---------------- factor implementations ----------------
def f_rel_mom(panel):
    mom = panel / panel.shift(5) - 1.0
    mom20 = mom.rolling(20).mean() * 20  # approx 20d momentum with 5d skip
    out = mom20.sub(mom20.median(axis=1), axis=0)
    return out


def f_beta_ew(panel):
    ret = panel.pct_change()
    ew = ret.mean(axis=1, skipna=True)
    cols = {}
    for a in panel.columns:
        cols[a] = rolling_beta(ret[a], ew, 60)
    return pd.DataFrame(cols, index=panel.index)


def f_downside_vol_ratio(panel):
    ret = panel.pct_change()
    cols = {}
    for a in panel.columns:
        s = ret[a]
        total = s.rolling(20).std()
        downside = s.where(s < 0).rolling(20).std()
        cols[a] = -(downside / total.replace(0, np.nan))
    return pd.DataFrame(cols, index=panel.index)


def f_max_ret(panel):
    ret = panel.pct_change()
    return ret.rolling(20).max()


def f_eurusd_beta_cond(panel, macro):
    eurusd = macro["EURUSD"]
    eurusd_ret = eurusd.pct_change()
    ret = panel.pct_change()
    cols = {}
    for a in panel.columns:
        b = rolling_beta(ret[a], eurusd_ret, 60)
        cols[a] = b * (eurusd / eurusd.shift(20) - 1.0)
    return pd.DataFrame(cols, index=panel.index)


def f_corr_ew(panel):
    ret = panel.pct_change()
    cols = {}
    for a in panel.columns:
        corrs = []
        for b in panel.columns:
            if b == a:
                continue
            c = ret[a].rolling(60).corr(ret[b])
            corrs.append(c)
        cols[a] = pd.concat(corrs, axis=1).mean(axis=1)
    return pd.DataFrame(cols, index=panel.index)


def f_kurt_skip5(panel):
    ret = panel.pct_change()
    return ret.rolling(20).kurt()


FACTORS = {
    "rel_mom_20d_skip5": (f_rel_mom, 1.0),
    "beta_ew_60d": (f_beta_ew, 1.0),
    "downside_vol_ratio_20": (f_downside_vol_ratio, 1.0),
    "max_ret_20d": (f_max_ret, 1.0),
    "eurusd_beta_cond_60x20": (f_eurusd_beta_cond, 1.0),
    "corr_ew_60": (f_corr_ew, 1.0),
    "kurt_20d_skip5": (f_kurt_skip5, 1.0),
}

results = {}
for fid, (fn, direction) in FACTORS.items():
    fdf = fn(panel) if fid != "eurusd_beta_cond_60x20" else fn(panel, macro)
    res = validate_factor(fdf, panel, h=10, direction=direction)
    res["decay"] = decay_profile(fdf, panel, direction=direction)
    res["turnover_10d_rank"] = turnover_10d(fdf, panel)
    res.update(coverage(fdf, panel))
    # direction flip check: use sign of h10 IC
    if res["ic"] < 0:
        res["flip_direction"] = -1.0
        res = validate_factor(fdf, panel, h=10, direction=-1.0)
        res["decay"] = decay_profile(fdf, panel, direction=-1.0)
        res["turnover_10d_rank"] = turnover_10d(fdf, panel)
        res.update(coverage(fdf, panel))
    else:
        res["flip_direction"] = 1.0
    results[fid] = res

print("\n=== RE-VALIDATION RESULTS (data thru 2027-03-01) ===")
print(f"Admission gates: |IC|>={ADMISSION['ic']}, |ICIR|>={ADMISSION['icir']} @h10")
for fid, r in results.items():
    per_year = r.get("per_year", {})
    py = ", ".join(f"{y}: ic={v['ic']:.4f}/icir={v['icir']:.3f}/n={v['n']}" for y, v in per_year.items())
    dec = ", ".join(f"h{h}={v:.4f}" for h, v in r["decay"].items())
    status = "PASS" if (abs(r["ic"]) >= ADMISSION["ic"] and abs(r["icir"]) >= ADMISSION["icir"]) else "FAIL"
    print(f"\n[{status}] {fid} (dir={r['flip_direction']})")
    print(f"  h10 IC={r['ic']:.5f} ICIR={r['icir']:.4f} hit={r['hit']:.3f} n={r['n']}")
    print(f"  decay: {dec}")
    print(f"  turnover_10d_rank={r['turnover_10d_rank']:.3f} "
          f"cov_asset_days={r['coverage_asset_days']:.3f} cov_dates_ge8={r['coverage_dates_ge8']:.3f}")
    print(f"  per-year: {py}")

with open("scripts/miner3_20270302_revalidation.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner3_20270302_revalidation.json")

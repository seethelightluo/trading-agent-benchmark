"""Re-validation of currently effective + excluded factors through 2035-12-05.

Checks full-period gate (|IC|>=0.0070, |ICIR|>=0.0840) plus recent 2y and 1y
windows to catch regime drift / sign flips.
"""
import sys, os, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner2_20351206_lib import (load_all_data, build_panel, forward_returns,
                                 ic_series, summarize_ic, coverage_turnover,
                                 beta_series)

data, macro = load_all_data()
print(f"Loaded {len(data)} instruments, thru {data['SPX'].index.max().date()}")
print(f"Date range: {data['SPX'].index.min().date()} .. {data['SPX'].index.max().date()}")

H = 10
fwd = forward_returns(data, H)


def wti_beta(sym, df):
    return beta_series(df["ret"], data["WTI"]["ret"], 60)


def cny_beta(sym, df):
    return beta_series(df["ret"], macro["USDCNY"]["ret"], 60)


def dxy_beta(sym, df):
    return beta_series(df["ret"], macro["DXY"]["ret"], 60)


def kurt_20(sym, df):
    r = df["ret"]
    mu = r.rolling(20, min_periods=8).mean()
    m2 = ((r - mu) ** 2).rolling(20, min_periods=8).mean()
    m4 = ((r - mu) ** 4).rolling(20, min_periods=8).mean()
    return m4 / (m2 ** 2) - 3.0


def mom_10d(sym, df):
    return df["close"].shift(5) / df["close"].shift(15) - 1.0


def mom_120d(sym, df):
    return df["close"].shift(5) / df["close"].shift(125) - 1.0


def semi_down_ratio(sym, df):
    r = df["ret"]
    down = np.minimum(r, 0.0) ** 2
    up = np.maximum(r, 0.0) ** 2
    return np.sqrt(down.rolling(20, min_periods=8).mean()) / \
           np.sqrt(up.rolling(20, min_periods=8).mean()) - 1.0


def tail_ratio(sym, df):
    r = df["ret"]
    q95 = r.rolling(20, min_periods=10).quantile(0.95)
    q05 = r.rolling(20, min_periods=10).quantile(0.05)
    return q95 / q05.abs()


def time_under_water(sym, df):
    c = df["close"]
    rollmax = c.rolling(120, min_periods=20).max()
    at_peak = (c >= rollmax * (1 - 1e-12)).astype(int)
    vals = []
    last_peak = np.nan
    for i, (dt, v) in enumerate(at_peak.items()):
        if v == 1:
            last_peak = i
            vals.append(0.0)
        else:
            vals.append(float(i - last_peak) if pd.notna(last_peak) else np.nan)
    return pd.Series(vals, index=c.index)


def trend_r2_signed(sym, df):
    lc = np.log(df["close"])
    t = np.arange(len(lc))
    ts = pd.Series(t, index=lc.index)
    cov = lc.rolling(30, min_periods=18).cov(ts)
    var_t = ts.rolling(30, min_periods=18).var()
    var_lc = lc.rolling(30, min_periods=18).var()
    r2 = cov ** 2 / (var_t * var_lc)
    slope = cov / var_t
    return np.sign(slope) * r2


def vix_beta_cond(sym, df):
    b = beta_series(df["ret"], macro["VIX"]["ret"], 60)
    vix_chg = macro["VIX"]["close"] / macro["VIX"]["close"].shift(20) - 1.0
    return -b * vix_chg


def vol_of_vol(sym, df):
    v = df["ret"].rolling(20, min_periods=10).std()
    return v.rolling(60, min_periods=20).std()


factors = {
    "cny_beta_60": cny_beta,
    "trend_r2_30_signed": trend_r2_signed,
    "mom_120d_skip5": mom_120d,
    "dxy_beta_60": dxy_beta,
    "vol_of_vol20x60": vol_of_vol,
    "time_under_water_120": time_under_water,
    "semi_down_ratio_20": semi_down_ratio,
    "vix_beta_cond_60x20": vix_beta_cond,
    "mom_10d_skip5": mom_10d,
    "tail_ratio_20": tail_ratio,
    "kurt_20": kurt_20,
    "WTI_BETA_60": wti_beta,
}

results = {"full": {}, "recent_2y": {}, "recent_1y": {}}
print("=== FULL PERIOD (2020..2035-12-05) H=10 ===")
for name, fn in factors.items():
    panel = build_panel(data, fn, min_valid=8)
    cov, turn = coverage_turnover(panel)
    ics = ic_series(panel, fwd, min_valid=8)
    s = summarize_ic(ics, name)
    results["full"][name] = {**s, "coverage": cov, "turnover": turn}
    gate = (abs(s["IC"]) >= 0.0070) and (abs(s["ICIR"]) >= 0.0840)
    print(f"{name:24s} IC={s['IC']:+.5f} ICIR={s['ICIR']:+.3f} hit={s['hit_ratio']:.3f} "
          f"cov={cov:.3f} turn={turn:.4f} n={s['n_dates']} GATE={'PASS' if gate else 'fail'}")

for wlabel, wstart in [("recent_2y", "2033-12-05"), ("recent_1y", "2034-12-05")]:
    print(f"\n=== {wlabel.upper()} (thru 2035-12-05) ===")
    for name, fn in factors.items():
        panel = build_panel(data, fn, min_valid=8)
        ics = ic_series(panel, fwd, min_valid=8)
        rec = ics[ics.index >= pd.Timestamp(wstart)]
        if len(rec) >= 30:
            s = summarize_ic(rec, name)
            results[wlabel][name] = s
            print(f"{name:24s} IC={s['IC']:+.5f} ICIR={s['ICIR']:+.3f} hit={s['hit_ratio']:.3f} n={s['n_dates']}")
        else:
            print(f"{name:24s} insufficient dates: {len(rec)}")

with open("scripts/miner2_20351206_reval_results.json", "w") as fp:
    json.dump(results, fp, indent=1, default=str)
print("\nsaved reval results")
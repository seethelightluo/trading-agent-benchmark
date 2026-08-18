"""Re-validation of all currently effective + previously excluded factors through 2029-11-28."""
import sys, os, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner2_20291129_lib import (load_all_data, build_panel, forward_returns,
                                 ic_series, summarize_ic, coverage_turnover,
                                 beta_series)

data, macro = load_all_data()
print(f"Loaded {len(data)} instruments, {len(macro)} macro series, thru {data['SPX'].index.max().date()}")
print(f"Date range: {data['SPX'].index.min().date()} .. {data['SPX'].index.max().date()}")

H = 10
fwd = forward_returns(data, H)


def wti_beta(sym, df):
    return beta_series(df["ret"], data["WTI"]["ret"], 60)


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
    out = pd.Series(np.nan, index=c.index)
    last_peak = np.nan
    vals = []
    for i, (dt, v) in enumerate(at_peak.items()):
        if v == 1:
            last_peak = i
            vals.append(0.0)
        else:
            vals.append(float(i - last_peak) if pd.notna(last_peak) else np.nan)
    out = pd.Series(vals, index=c.index)
    return out


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
    "trend_r2_30_signed": trend_r2_signed,
    "semi_down_ratio_20": semi_down_ratio,
    "vol_of_vol20x60": vol_of_vol,
    "mom_120d_skip5": mom_120d,
    "time_under_water_120": time_under_water,
    "vix_beta_cond_60x20": vix_beta_cond,
    "dxy_beta_60": dxy_beta,
    "mom_10d_skip5": mom_10d,
    "tail_ratio_20": tail_ratio,
    "kurt_20": kurt_20,
    "WTI_BETA_60": wti_beta,
}

results = {}
for name, fn in factors.items():
    panel = build_panel(data, fn, min_valid=8)
    cov, turn = coverage_turnover(panel)
    ics = ic_series(panel, fwd, min_valid=8)
    s = summarize_ic(ics, name)
    results[name] = {**s, "coverage": cov, "turnover": turn}
    print(f"{name:24s} IC={s['IC']:+.5f} ICIR={s['ICIR']:+.3f} hit={s['hit_ratio']:.3f} "
          f"cov={cov:.3f} turn={turn:.4f} n_dates={s['n_dates']}")

print("\n--- GATE CHECK (|IC|>=0.0070, |ICIR|>=0.0840) ---")
for name, r in results.items():
    gate = (abs(r["IC"]) >= 0.0070) and (abs(r["ICIR"]) >= 0.0840)
    print(f"{name:24s} PASS={gate}  IC={r['IC']:+.5f} ICIR={r['ICIR']:+.3f}")

# recent-window check (last ~2 years) to catch drift
print("\n--- RECENT WINDOW (2027-11-28..2029-11-28) ---")
for name, fn in factors.items():
    panel = build_panel(data, fn, min_valid=8)
    ics = ic_series(panel, fwd, min_valid=8)
    rec = ics[(ics.index >= pd.Timestamp("2027-11-28"))]
    if len(rec) >= 30:
        s = summarize_ic(rec, name)
        print(f"{name:24s} IC={s['IC']:+.5f} ICIR={s['ICIR']:+.3f} hit={s['hit_ratio']:.3f} n={s['n_dates']}")
    else:
        print(f"{name:24s} insufficient recent dates: {len(rec)}")

with open("scripts/miner2_20291129_reval_results.json", "w") as fp:
    json.dump({k: {kk: (str(vv) if hasattr(vv, 'isoformat') else vv) for kk, vv in v.items()} for k, v in results.items()}, fp, indent=1, default=str)
print("saved reval results")

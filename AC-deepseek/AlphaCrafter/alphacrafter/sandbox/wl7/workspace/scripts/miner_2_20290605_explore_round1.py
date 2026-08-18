"""miner_2 factor exploration round 1, through 2029-06-04.

Regime: deep bear (VIX ~88 EXTREME re-escalating), DXY fading (-2.3%/20d),
EURUSD +2.5%/20d, 4/15 frozen feeds (NDX/CN10Y/SOX/000688.SH).
Explore candidates orthogonal to the 8 active library factors.

Gate (h10 paper): abs(IC) >= 0.0070, abs(ICIR) >= 0.0840.
Correlation conflict threshold vs active library: 0.50.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, ACTIVE_LIB)

END = "2029-06-04"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
lib = library_panel(close, macro)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]} active_lib={list(ACTIVE_LIB.keys())}")


def max_lib_corr_pairs(cand):
    flat = cand.stack()
    pairs = {}
    for name, p in lib.items():
        pflat = p.reindex(cand.index).stack()
        df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        pairs[name] = float(df["f"].corr(df["p"]))
    best = max((abs(v), k) for k, v in pairs.items()) if pairs else (0.0, None)
    return best, pairs


def summarize(cand, name):
    fwd = forward_ret(close, 10)
    ic = daily_ic(cand, fwd)
    st = ic_stats(ic, 10)
    cov = coverage_stats(cand, fwd)
    turn = rank_turnover(cand, 10)
    # recent windows
    c500 = cand.tail(500)
    ic500 = daily_ic(c500, forward_ret(close, 10).reindex(c500.index))
    st500 = ic_stats(ic500, 10)
    c250 = cand.tail(250)
    ic250 = daily_ic(c250, forward_ret(close, 10).reindex(c250.index))
    st250 = ic_stats(ic250, 10)
    # decay
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        decay[h] = round(ic_stats(daily_ic(cand, forward_ret(close, h)), h)["ic"], 4)
    best, pairs = max_lib_corr_pairs(cand)
    return dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                ic500=st500["ic"], icir500=st500["icir"], n500=st500["n"],
                ic250=st250["ic"], icir250=st250["icir"], n250=st250["n"],
                covAD=cov["coverage_asset_days"], covD8=cov["coverage_dates_ge8"], turn=turn,
                decay=decay, maxcorr=best, pairs=pairs)


# ---------------- candidate constructions ----------------
cands = {}

# A. skew_20d_skip5 : rolling skewness of daily returns (skip 5)
r_sk = ret.shift(5)
cands["skew_20d_skip5"] = r_sk.rolling(20, min_periods=12).skew()

# B. drawdown_60d : 1 - close/rolling_max(close,60)
cands["drawdown_60d"] = 1.0 - close / close.rolling(60, min_periods=30).max()

# C. close_pos_20d : (close - min(low,20)) / (max(high,20) - min(low,20))
lo = close.copy()
hi = close.copy()
for a in close.columns:
    raw = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").reindex(close.index)
    lo[a] = raw["low"].ffill()
    hi[a] = raw["high"].ffill()
rng = hi.rolling(20, min_periods=10).max() - lo.rolling(20, min_periods=10).min()
cands["close_pos_20d"] = (close - lo.rolling(20, min_periods=10).min()) / rng.replace(0, np.nan)

# D. yield_beta_cond_60x20 : beta(asset_ret, US10Y yield chg, 60) * US10Y 20d chg
us10y = close["US10Y"]
y_r = us10y.pct_change()
cov = ret.rolling(60, min_periods=30).cov(y_r)
var = y_r.rolling(60, min_periods=30).var()
beta_y = cov.divide(var, axis=0)
y_mom = us10y / us10y.shift(20) - 1.0
cands["yield_beta_cond_60x20"] = beta_y.multiply(y_mom, axis=0)

# E. usdjpy_cond_60x20 : beta(asset_ret, USDJPY ret, 60) * USDJPY 20d chg
jpy = macro["USDJPY"]
j_r = jpy.pct_change()
covj = ret.rolling(60, min_periods=30).cov(j_r)
varj = j_r.rolling(60, min_periods=30).var()
beta_j = covj.divide(varj, axis=0)
j_mom = jpy / jpy.shift(20) - 1.0
cands["usdjpy_cond_60x20"] = beta_j.multiply(j_mom, axis=0)

# F. vix_z_beta_60x20 : beta(asset_ret, VIX ret, 60) * (VIX / VIX.rolling(60).mean() - 1)
vix = macro["VIX"]
v_r = vix.pct_change()
covv = ret.rolling(60, min_periods=30).cov(v_r)
varv = v_r.rolling(60, min_periods=30).var()
beta_v = covv.divide(varv, axis=0)
vix_z = vix / vix.rolling(60, min_periods=30).mean() - 1.0
cands["vix_z_beta_60x20"] = beta_v.multiply(vix_z, axis=0)

# G. vol_expansion_20x60 : vol20/vol60 - 1
vol20 = ret.rolling(20, min_periods=12).std()
vol60 = ret.rolling(60, min_periods=30).std()
cands["vol_expansion_20x60"] = vol20 / vol60 - 1.0

# H. volume_trend_20x60 : mean volume 20d / mean volume 60d - 1
vol_panel = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
for a in close.columns:
    raw = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.set_index("date").reindex(close.index)
    v = raw["volume"].ffill()
    vol_panel[a] = v
v20 = vol_panel.rolling(20, min_periods=10).mean()
v60 = vol_panel.rolling(60, min_periods=30).mean()
cands["volume_trend_20x60"] = v20 / v60 - 1.0

# ---------------- summary ----------------
print(f"\n{'candidate':24s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} | {'IC500':>7s} {'ICIR500':>7s} | {'IC250':>7s} {'ICIR250':>7s} | {'covAD':>6s} {'covD8':>5s} {'turn':>6s} {'maxRho':>7s}")
rows = []
for name, cand in cands.items():
    s = summarize(cand, name)
    rows.append(s)
    print(f"{name:24s} {s['ic']:7.4f} {s['icir']:7.3f} {s['hit']:5.2f} {s['n']:5d} | "
          f"{s['ic500']:7.4f} {s['icir500']:7.3f} | {s['ic250']:7.4f} {s['icir250']:7.3f} | "
          f"{s['covAD']:6.2f} {s['covD8']:5.2f} {s['turn']:6.2f} {s['maxcorr'][0]:7.3f}")

print("\nDecay IC by horizon + pairwise corr vs active library:")
for s in rows:
    dec = " ".join(f"h{h}:{v:+.3f}" for h, v in s["decay"].items())
    top = sorted(s["pairs"].items(), key=lambda kv: -abs(kv[1]))[:4]
    pair_s = " ".join(f"{k}:{v:+.2f}" for k, v in top)
    print(f"{s['name']:24s} {dec}")
    print(f"{'':24s} corr: {pair_s}")

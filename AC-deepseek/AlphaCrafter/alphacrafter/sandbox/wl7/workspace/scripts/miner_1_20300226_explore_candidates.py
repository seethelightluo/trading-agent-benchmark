"""miner_1 broad exploration of candidate factor families at 2030-02-25.

Screens multiple interpretable factor ideas; each is a single construction
family. Follow-up focused validation scripts will be written for promising ones.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, ACTIVE_LIB)

END = "2030-02-25"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
fwd10 = forward_ret(close, 10)
lib_panels = library_panel(close, macro)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

def ic_recent(panel, window):
    sub = panel.tail(window)
    ic = daily_ic(sub, fwd10.reindex(sub.index))
    return ic_stats(ic, 10)

def per_year(panel):
    ic = daily_ic(panel, fwd10)
    out = []
    for yr in range(2025, 2031):
        s = ic_stats(ic.loc[ic.index.year == yr], 10)
        out.append(f"{yr}:{s['ic']:+.3f}/{s['icir']:+.2f}")
    return "  ".join(out)

def max_lib_corr(panel):
    flat = panel.stack()
    best = 0.0
    for name, p in lib_panels.items():
        pf = p.reindex(panel.index).stack()
        df = pd.concat([flat.rename("f"), pf.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        rho = abs(float(df["f"].corr(df["p"])))
        if rho > best:
            best = rho
    return best

cands = {}

# A. USDJPY beta * USDJPY momentum (macro-conditional, like dxy/eurusd)
def cand_usdjpy(close, macro, beta_win=60, cond_win=20, min_periods=30):
    r = close.pct_change()
    fx = macro["USDJPY"].pct_change()
    beta = r.rolling(beta_win, min_periods=min_periods).cov(fx) / fx.rolling(beta_win, min_periods=min_periods).var()
    mom = macro["USDJPY"] / macro["USDJPY"].shift(cond_win) - 1.0
    return beta.multiply(mom, axis=0)
cands["usdjpy_beta_cond_60x20"] = cand_usdjpy(close, macro)

# B. USDCNY beta * USDCNY momentum
def cand_usdcny(close, macro, beta_win=60, cond_win=20, min_periods=30):
    r = close.pct_change()
    fx = macro["USDCNY"].pct_change()
    beta = r.rolling(beta_win, min_periods=min_periods).cov(fx) / fx.rolling(beta_win, min_periods=min_periods).var()
    mom = macro["USDCNY"] / macro["USDCNY"].shift(cond_win) - 1.0
    return beta.multiply(mom, axis=0)
cands["usdcny_beta_cond_60x20"] = cand_usdcny(close, macro)

# C. realized skewness 20d skip5
def cand_skew(close, window=20, skip=5, min_periods=12):
    r = close.pct_change().shift(skip)
    return r.rolling(window, min_periods=min_periods).skew()
cands["skew_20d_skip5"] = cand_skew(close)

# D. worst daily return over 20d (min_ret)
def cand_min_ret(close, window=20):
    return close.pct_change().rolling(window).min()
cands["min_ret_20d"] = cand_min_ret(close)

# E. trend position: close/MA60 - 1 (cross-sectional)
def cand_trend_pos(close, window=60):
    return close / close.rolling(window).mean() - 1.0
cands["trend_pos_60d"] = cand_trend_pos(close)

# G. serial correlation lag-1 over 60d
def cand_autocorr(close, window=60, min_periods=30):
    r = close.pct_change()
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for a in r.columns:
        x = r[a]
        m1 = x.rolling(window, min_periods=min_periods).mean()
        v = x.rolling(window, min_periods=min_periods).var()
        l1 = x.shift(1)
        c = ((x - m1) * (l1 - m1.shift(1))).rolling(window, min_periods=min_periods).mean()
        out[a] = c / v
    return out
cands["autocorr_60d"] = cand_autocorr(close)

# H. range position (stochastic): (close - low20)/(high20-low20)
def cand_range_pos(close, window=20):
    hi = close.rolling(window).max()
    lo = close.rolling(window).min()
    return (close - lo) / (hi - lo)
cands["range_pos_20d"] = cand_range_pos(close)

# J. momentum acceleration: mom20 - mom60
def cand_mom_accel(close, fast=20, slow=60):
    return (close / close.shift(fast) - 1.0) - (close / close.shift(slow) - 1.0)
cands["mom_accel_20x60"] = cand_mom_accel(close)

# F. vol acceleration: vol20/vol60 - 1
def cand_vol_accel(close, fast=20, slow=60, min_periods=30):
    vf = close.pct_change().rolling(fast).std()
    vs = close.pct_change().rolling(slow, min_periods=min_periods).std()
    return vf / vs - 1.0
cands["vol_accel_20x60"] = cand_vol_accel(close)

# O. RSI-14 cross-sectional
def cand_rsi(close, window=14):
    r = close.pct_change()
    up = r.clip(lower=0).rolling(window).mean()
    dn = (-r.clip(upper=0)).rolling(window).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)
cands["rsi_14d"] = cand_rsi(close)

print(f"\n{'factor':26s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} | {'IC500':>7s} {'ICIR500':>7s} | {'IC250':>7s} {'ICIR250':>7s} | {'covAD':>6s} {'turn':>6s} {'maxLib':>6s}")
rows = []
for name, f in cands.items():
    st = ic_stats(daily_ic(f, fwd10), 10)
    s500 = ic_recent(f, 500)
    s250 = ic_recent(f, 250)
    cov = coverage_stats(f, fwd10)
    turn = rank_turnover(f, 10)
    mlc = max_lib_corr(f)
    rows.append((name, st, s500, s250, cov, turn, mlc))
    print(f"{name:26s} {st['ic']:7.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} | "
          f"{s500['ic']:7.4f} {s500['icir']:7.3f} | {s250['ic']:7.4f} {s250['icir']:7.3f} | "
          f"{cov['coverage_asset_days']:6.2f} {turn:6.2f} {mlc:6.3f}")

print("\nGATE check full-window h10 (abs IC>=0.0070, abs ICIR>=0.0840):")
for name, st, s500, s250, cov, turn, mlc in rows:
    gp = abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840
    recent_ok = abs(s250["ic"]) >= 0.0070 and abs(s250["icir"]) >= 0.0840
    corr_ok = mlc < 0.5
    print(f"{name:26s} IC={st['ic']:+.4f} ICIR={st['icir']:+.3f} gate={'PASS' if gp else 'FAIL'} "
          f"recent250={'OK' if recent_ok else 'weak'} maxLib={mlc:.3f} {'CORR-OK' if corr_ok else 'CORR-HIGH'}")

print("\nPer-year h10 IC/ICIR (2025..2030):")
for name, f in cands.items():
    print(f"{name:26s} " + per_year(f))

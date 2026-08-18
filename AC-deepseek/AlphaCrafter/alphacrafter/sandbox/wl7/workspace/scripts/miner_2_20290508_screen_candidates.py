"""miner_2 exploration round 2029-05-08: novel downside-risk / defensive factor candidates.

Regime context: deep bear tape (VIX ~73 EXTREME, 9 consecutive down blocks before a
flat block). Existing active library already covers rel_mom, beta_ew, corr_ew,
downside_vol_ratio, kurt, max_ret, dxy/eurusd beta-cond. Candidates below are NEW
variants emphasizing downside risk / defensiveness / trend quality that are not in
the active library and were not evicted before.

Admission gates (15-instrument universe, horizon 10):
  abs IC >= 0.0070, abs ICIR >= 0.0840
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (
    load_close, load_macro, forward_ret, daily_ic, ic_stats, summarize,
    rank_turnover, coverage_stats, library_panel, max_lib_corr,
)

END = "2029-05-08"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
mkt = ret.mean(axis=1)

print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}")

# ---------------- candidate factor panels ----------------
cands = {}

# 1. skew_20d_skip5: rolling skewness of 5d-lagged returns (tail asymmetry, distinct from kurtosis)
cands["skew_20d_skip5"] = ret.shift(5).rolling(20, min_periods=12).skew()

# 2. drawdown_60d: distance from 60d rolling high (defensive = small drawdown)
cands["drawdown_60d"] = close / close.rolling(60, min_periods=30).max() - 1.0

# 3. downside_beta_60d: beta vs EW market computed only on down-market days (crash sensitivity), sign-flipped
def downside_beta(close, mkt, window=60, min_periods=30):
    r = close.pct_change()
    down = mkt < 0
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        ra = r[a]
        # rolling stats restricted to down days: use product trick
        ra_d = ra.where(down)
        rm_d = mkt.where(down)
        # cov via E[xy]-E[x]E[y] on down days, renormalized by down-day count
        n = down.rolling(window, min_periods=min_periods).sum()
        exy = (ra_d * rm_d).rolling(window, min_periods=min_periods).sum() / n
        ex = ra_d.rolling(window, min_periods=min_periods).mean()
        ey = rm_d.rolling(window, min_periods=min_periods).mean()
        cov = exy - ex * ey
        var = (rm_d ** 2).rolling(window, min_periods=min_periods).mean() - ey ** 2
        out[a] = cov / var
    return out

cands["downside_beta_60d"] = -downside_beta(close, mkt, 60)

# 4. sharpe_ratio_20d: risk-adjusted momentum (mean/std of daily ret over 20d)
cands["sharpe_ratio_20d"] = ret.rolling(20, min_periods=12).mean() / ret.rolling(20, min_periods=12).std()

# 5. rsq_trend_60d: R^2 of log price vs linear time over 60d (trend quality/consistency)
def rsq_trend(close, window=60, min_periods=30):
    logp = np.log(close)
    t = np.arange(len(close))
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for a in close.columns:
        s = logp[a]
        out[a] = s.rolling(window, min_periods=min_periods).apply(
            lambda y: np.corrcoef(y, t[: len(y)])[0, 1] ** 2 if len(y) > 3 else np.nan,
            raw=True,
        )
    return out

cands["rsq_trend_60d"] = rsq_trend(close, 60)

# 6. worst_day_20d: worst daily return over 20d (tail risk), sign-flipped (higher = better tail)
cands["worst_day_20d"] = -ret.rolling(20, min_periods=12).min()

# 7. gain_loss_asym_20d: sum(pos ret)/|sum(neg ret)| over 20d (upside participation vs downside)
pos = ret.clip(lower=0)
neg = ret.clip(upper=0)
cands["gain_loss_asym_20d"] = pos.rolling(20, min_periods=12).sum() / neg.rolling(20, min_periods=12).sum().abs()

# ---------------- validation ----------------
lib_panels = library_panel(close, macro)
print("\n=== candidate validation (admission horizon H=10, period through END) ===")
print(f"{'factor':<24}{'IC_10':>9}{'ICIR_10':>9}{'hit_10':>7}{'n_10':>7}{'IC_5':>8}{'IC_20':>8}{'covAD':>7}{'turn':>7}{'maxLibRho':>10}")
rows = []
for name, f in cands.items():
    summ = summarize(f, close, horizons=(5, 10, 20))
    s10 = summ[10]
    cov = coverage_stats(f, forward_ret(close, 10))
    turn = rank_turnover(f, window=10)
    rho, pairs = max_lib_corr(f, lib_panels)
    ic5 = summ[5]["ic"] if summ[5]["n"] else np.nan
    ic20 = summ[20]["ic"] if summ[20]["n"] else np.nan
    print(f"{name:<24}{s10['ic']:>9.4f}{s10['icir']:>9.4f}{s10['hit']:>7.3f}{s10['n']:>7d}"
          f"{ic5:>8.4f}{ic20:>8.4f}{cov['coverage_asset_days']:>7.3f}{turn:>7.2f}{rho:>10.4f}")
    rows.append((name, s10, cov, turn, rho, pairs))

print("\n=== pairwise rho detail (vs active library) for near-gate candidates ===")
for name, s10, cov, turn, rho, pairs in rows:
    if abs(s10["ic"]) >= 0.005 or abs(s10["icir"]) >= 0.06:
        print(name, pairs)

print("\n=== full decay for candidates with abs(IC_10)>=0.005 ===")
for name, s10, cov, turn, rho, pairs in rows:
    if abs(s10["ic"]) >= 0.005:
        summ = summarize(cands[name], close)
        decay = {str(h): round(summ[h]["ic"], 4) for h in (1, 2, 3, 5, 10, 20)}
        print(name, "decay_ic:", decay)

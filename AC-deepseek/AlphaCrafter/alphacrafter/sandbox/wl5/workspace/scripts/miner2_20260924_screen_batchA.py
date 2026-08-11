"""miner_2 screen 2026-09-24 cycle: novel cross-asset factor families (batch A).

Avoids library (trend_r2, semi_down_ratio, mom_120/10, vol_of_vol, dxy_beta,
time_under_water, kurt_20, vix_beta_cond, WTI_BETA) and recent screens by
miner_1 (rates_beta, mkt_beta, alpha_mom, rel_mom, up_day_ratio, parkinson,
day_eff, vol_mom, max_up, hi_lo_pos) and miner_3 (fx_beta, downside_beta_asym,
amihud, range_pos, tail_ratio, skew_term, gap_ratio, xau_beta, updown_vol_asym,
zscore_60, ret_autocorr).

Candidates:
  eff_ratio_10      - Kaufman efficiency ratio 10d (trend quality via path length)
  ma_cross_10x30    - normalized fast/slow MA crossover
  ret_autocorr_10   - 10d autocorrelation of daily returns (persistence)
  vol_cluster_20    - corr(r^2, r^2_lag1) over 20d (vol clustering)
  gk_vol_term_10x60 - Garman-Klass vol term structure 10d/60d
  dist_ma60         - (close/SMA60 - 1) medium-term trend position
  downside_mean_10  - mean of negative daily returns over 10d (down-move size)
  dd_depth_60       - current drawdown depth from 60d max
  corr_change_10x60 - 10d vs 60d correlation with EW composite (regime shift)
  range_amp_20      - mean (H-L)/C over 20d (intraday amplitude)
  cn10y_beta_60     - beta of asset returns to CN10Y returns
  corr_us10y_10     - 10d correlation with US10Y returns
  vol_ratio_5x60    - volume 5d/60d ratio (liquidity/participation shift)
  sharpe_60         - 60d risk-adjusted return (mean/std)

Gate: |IC|>=0.007 and |ICIR|>=0.084 at h=10 on >=8 valid instruments.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from miner3_lib import load_close_panel, rank_ic

VIS = '2026-09-23'
H = 10
C, V, H_, L, O = load_close_panel(days=4000)
R = C.pct_change()
LR = np.log(C).diff()
print(f"panel: dates={C.index.min().date()}..{C.index.max().date()} rows={len(C)} assets={len(C.columns)}", flush=True)

mkt = R.mean(axis=1)  # equal-weight composite of the 15 tradable assets

# ---- macro panels for conditional candidates ----
def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    return df['close'].reindex(C.index).ffill()

CN10Y = C['CN10Y']
US10Y = C['US10Y']

def rolling_beta(y, x, win, minp=30):
    cov = y.rolling(win, min_periods=minp).cov(x)
    var = x.rolling(win, min_periods=minp).var()
    return (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

# 1. Kaufman efficiency ratio
cands = {'eff_ratio_10': ((C - C.shift(10)).abs() / C.diff().abs().rolling(10).sum()).replace([np.inf, -np.inf], np.nan)}

# 2. MA crossover
ma10 = C.rolling(10).mean()
ma30 = C.rolling(30).mean()
cands['ma_cross_10x30'] = ((ma10 - ma30) / ma30.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

# 3. return autocorrelation 10d
cands['ret_autocorr_10'] = R.rolling(10).corr(R.shift(1))

# 4. vol clustering 20d
rsq = R ** 2
cands['vol_cluster_20'] = rsq.rolling(20).corr(rsq.shift(1))

# 5. Garman-Klass vol term structure
gk = np.sqrt(0.5 * (np.log(H_ / L) ** 2) - (2 * np.log(2) - 1) * (np.log(C / O) ** 2))
gk10 = gk.rolling(10).mean()
gk60 = gk.rolling(60).mean()
cands['gk_vol_term_10x60'] = (gk10 / gk60.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

# 6. distance from 60d MA
cands['dist_ma60'] = ((C - C.rolling(60).mean()) / C.rolling(60).mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

# 7. mean negative daily return over 10d
cands['downside_mean_10'] = R.where(R < 0).rolling(10).mean()

# 8. drawdown depth from 60d max
cands['dd_depth_60'] = (C / C.rolling(60).max() - 1.0)

# 9. correlation regime shift with EW composite
c10 = R.rolling(10).corr(mkt)
c60 = R.rolling(60).corr(mkt)
cands['corr_change_10x60'] = (c10 - c60)

# 10. intraday amplitude
cands['range_amp_20'] = ((H_ - L) / C.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).rolling(20).mean()

# 11. beta to CN10Y returns
cn10y_ret = CN10Y.pct_change()
cands['cn10y_beta_60'] = rolling_beta(R, cn10y_ret, 60)

# 12. 10d correlation with US10Y returns
us10y_ret = US10Y.pct_change()
cands['corr_us10y_10'] = R.rolling(10).corr(us10y_ret)

# 13. volume ratio 5d/60d
cands['vol_ratio_5x60'] = (V.rolling(5).mean() / V.rolling(60).mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

# 14. 60d Sharpe
cands['sharpe_60'] = (R.rolling(60).mean() / R.rolling(60).std().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

FR = R.shift(-H)
print(f"\n{'factor':<22}{'IC':>8}{'ICIR':>8}{'hit':>7}{'n':>6}{'cov':>6}  regime(20-22/23-24/25-26)", flush=True)
rows = []
for name, fp in cands.items():
    if fp is None or fp.shape[0] == 0:
        continue
    s = rank_ic(fp, FR)
    if s is None or len(s) < 30:
        print(f"{name:<22} insufficient dates ({0 if s is None else len(s)})", flush=True)
        continue
    ic = s.mean()
    icir = ic / s.std() if s.std() > 0 else 0.0
    hit = (s > 0).mean()
    cov = float(fp.notna().sum().sum()) / float(fp.size)
    regs = []
    for lo, hi in [("2020-01-01", "2022-12-31"), ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]:
        sub = s[(s.index >= lo) & (s.index <= hi)]
        regs.append(round(sub.mean(), 3) if len(sub) >= 20 else float('nan'))
    rows.append((name, ic, icir, hit, len(s), cov, regs))
    flag = '  <== PASS' if (abs(ic) >= 0.007 and abs(icir) >= 0.084) else ''
    print(f"{name:<22}{ic:>8.4f}{icir:>8.4f}{hit:>7.3f}{len(s):>6}{cov:>6.2f}  {regs[0]}/{regs[1]}/{regs[2]}{flag}", flush=True)

print("\n--- Top by |IC|*|ICIR| ---", flush=True)
rows.sort(key=lambda r: abs(r[1] * r[2]), reverse=True)
for r in rows[:10]:
    print(f"{r[0]:<22} IC={r[1]:.4f} ICIR={r[2]:.4f} hit={r[3]:.3f} n={r[4]} cov={r[5]:.2f} regime={r[6]}", flush=True)

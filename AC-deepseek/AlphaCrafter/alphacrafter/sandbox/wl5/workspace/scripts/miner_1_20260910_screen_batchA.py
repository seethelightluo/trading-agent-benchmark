"""miner_1 screen 2026-09-10 cycle: novel cross-asset factor families.

Candidates (all avoid duplication with library: trend_r2, semi_down_ratio, mom_120/10,
vol_of_vol, dxy_beta, time_under_water, kurt_20, vix_beta_cond, WTI_BETA, and prior
screens by miner_2/miner_3):
  1. rates_beta_60      - rolling beta of asset returns to US10Y daily returns
  2. mkt_beta_60        - rolling beta to equal-weight cross-asset composite
  3. alpha_mom_60       - idiosyncratic momentum: trailing 60d cumulative OLS alpha vs market
  4. rel_mom_20         - 20d momentum minus cross-sectional mean (relative strength)
  5. up_day_ratio_20    - fraction of up days over trailing 20d (path quality)
  6. parkinson_ratio_20 - Parkinson (range) vol / close-to-close vol over 20d
  7. day_eff_20         - mean |daily ret| / intraday (high-low) range (directional efficiency)
  8. vol_mom_60         - 20d realized vol now vs 60d ago (vol trend)
  9. max_up_20          - max daily return over 20d (positive tail exposure)
 10. hi_lo_pos_20       - (close - min(low,20)) / (max(high,20)-min(low,20)) stochastic position

Gate: |IC|>=0.007 and |ICIR|>=0.084 at h=10 on >=8 valid instruments.
"""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_validate import load_panel, closes_panel, forward_returns, ic_series, summary_metrics

VIS = '2026-09-09'  # last completed trading day before current date 2026-09-10
H = 10
close = closes_panel(VIS)
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS}", flush=True)

src = load_panel(visible_through=VIS, source='stock')
opens = pd.DataFrame({s: df.set_index('date')['open'].astype(float) for s, df in src.items()}).sort_index()
highs = pd.DataFrame({s: df.set_index('date')['high'].astype(float) for s, df in src.items()}).sort_index()
lows = pd.DataFrame({s: df.set_index('date')['low'].astype(float) for s, df in src.items()}).sort_index()
opens = opens.reindex(columns=close.columns)
highs = highs.reindex(columns=close.columns)
lows = lows.reindex(columns=close.columns)

ret = close.pct_change()
lret = np.log(close).diff()

# cross-asset composite (equal-weight of the 15 tradable assets)
mkt = ret.mean(axis=1)


def rolling_beta(y, x, win, minp):
    cov = y.rolling(win, min_periods=minp).cov(x)
    var = x.rolling(win, min_periods=minp).var()
    return (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


# 1. rates beta: beta of asset daily returns to US10Y daily returns
us10y = close['US10Y']
us10y_ret = us10y.pct_change()
rates_beta = rolling_beta(ret, us10y_ret, 60, 30)
candidates = {'rates_beta_60': rates_beta}

# 2. market beta to EW composite
candidates['mkt_beta_60'] = rolling_beta(ret, mkt, 60, 30)

# 3. idiosyncratic momentum: 60d cumulative alpha from market model
beta = rolling_beta(ret, mkt, 60, 30)
alpha = ret.mean(axis=0) - beta * mkt.rolling(60, min_periods=30).mean()
candidates['alpha_mom_60'] = alpha.rolling(60, min_periods=30).sum()

# 4. relative momentum 20d (cross-sectionally demeaned)
mom20 = close / close.shift(20) - 1.0
candidates['rel_mom_20'] = mom20 - mom20.mean(axis=1).to_frame('m').values

# 5. up-day ratio over 20d
candidates['up_day_ratio_20'] = (ret > 0).rolling(20, min_periods=10).mean()

# 6. Parkinson ratio: range vol vs close-to-close vol over 20d
hl = np.log(highs / lows)
park_vol = np.sqrt((hl ** 2).rolling(20, min_periods=10).mean() / (4.0 * np.log(2.0)))
cc_vol = lret.rolling(20, min_periods=10).std()
candidates['parkinson_ratio_20'] = (park_vol / cc_vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

# 7. intraday directional efficiency: |daily ret| / (high-low)/close, averaged over 20d
rng = (highs - lows) / close.replace(0, np.nan)
deff = (lret.abs() / rng.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
candidates['day_eff_20'] = deff.rolling(20, min_periods=10).mean()

# 8. vol momentum: 20d realized vol now vs 60d ago
v20 = ret.rolling(20, min_periods=10).std()
candidates['vol_mom_60'] = v20 / v20.shift(60) - 1.0

# 9. max daily return over 20d (positive tail)
candidates['max_up_20'] = ret.rolling(20, min_periods=10).max()

# 10. 20d stochastic position
hi20 = highs.rolling(20, min_periods=10).max()
lo20 = lows.rolling(20, min_periods=10).min()
candidates['hi_lo_pos_20'] = ((close - lo20) / (hi20 - lo20).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

fr = forward_returns(close, H)
out = {}
for name, sig in candidates.items():
    sig = sig.reindex(close.index)
    ic_s = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic_s, sig, fr, close, h=H)
    if m is None:
        print(f"{name}: INSUFFICIENT dates (n_ic={len(ic_s)})", flush=True)
        continue
    # regime split (custom, through VIS)
    regs = {}
    for rname, lo, hi in [("2020-2022", "2020-01-01", "2022-12-31"),
                          ("2023-2024", "2023-01-01", "2024-12-31"),
                          ("2025-2026", "2025-01-01", VIS)]:
        s = ic_s[(ic_s.index >= pd.Timestamp(lo)) & (ic_s.index <= pd.Timestamp(hi))].dropna()
        if len(s) >= 20:
            std = s.std(ddof=1)
            regs[rname] = {"ic": round(float(s.mean()), 4),
                           "icir": round(float(s.mean() / std), 4) if std > 0 else None,
                           "n": int(len(s))}
    m['regime'] = regs
    gate = bool(abs(m['ic']) >= 0.007 and abs(m['icir'] or 0) >= 0.084)
    out[name] = m
    print(f"=== {name}: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
          f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} turn={m['turnover_10d_rank']} "
          f"GATE={gate}", flush=True)
    print(f"    decay: {m['decay_ic_by_horizon']}", flush=True)
    print(f"    regime: {regs}", flush=True)

with open('scripts/miner_1_20260910_screen_batchA_results.json', 'w') as f:
    json.dump({'visible_through': VIS, 'horizon': H, 'candidates': out}, f, indent=1, default=str)
print("saved scripts/miner_1_20260910_screen_batchA_results.json", flush=True)

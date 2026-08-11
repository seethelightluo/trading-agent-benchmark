"""miner_1 2027-02-19 cycle: explore candidate factor families (batch 1).

Rationale (from trader feedback 20270219): commodity complex drag persists in a
side/bearish regime; reversal/vol factors still negative; momentum anchor helped
but top pick (COPPER) failed. We want trend-quality, macro-sensitivity
(USD/VIX), and crash-risk factors to diversify the reversal-heavy library.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner1_20270219_lib import (load_close_panel, load_macro, run_validation)

close = load_close_panel(days=2400)
macro = load_macro(days=2400)
ret = close.pct_change()

print(f"close panel: {close.shape}  {close.index[0].date()} -> {close.index[-1].date()}")
print(f"macro panel: {macro.shape}")

dxy_ret = macro["DXY"].pct_change()
vix_ret = macro["VIX"].pct_change()

# ---- 1. DXY beta 60d (USD sensitivity) ----
def rolling_beta(x, m, window):
    """Rolling beta of asset returns x vs market/macro returns m."""
    cov = x.rolling(window).cov(m)
    var = m.rolling(window).var()
    return cov / var

f_dxy_beta = rolling_beta(ret, dxy_ret, 60).reindex(close.index)

# ---- 2. Risk-adjusted momentum 120d (skip 5) / vol 60d ----
vol60 = ret.rolling(60).std()
mom120 = close.shift(5) / close.shift(125) - 1.0
f_ram = (mom120 / vol60).reindex(close.index)

# ---- 3. Distance from 250d high (drawdown depth) ----
f_d250 = (close / close.rolling(250).max() - 1.0)

# ---- 4. Trend win rate 60d (share of up days) ----
f_win = (ret > 0).rolling(60).mean()

# ---- 5. Skewness 60d (crash risk) ----
f_skew = ret.rolling(60).skew()

# ---- 6. Unconditional VIX beta 60d ----
f_vixb = rolling_beta(ret, vix_ret, 60).reindex(close.index)

# ---- 7. Trend R2 60d (trend clarity) ----
def trend_r2(s, window=60):
    x = np.arange(window)
    out = s.copy() * np.nan
    vals = s.values
    for i in range(window - 1, len(s)):
        y = vals[i - window + 1:i + 1]
        if np.isnan(y).any():
            continue
        if np.nanstd(y) == 0:
            continue
        b, a = np.polyfit(x, y, 1)
        yhat = a + b * x
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        out.iloc[i] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return out

f_r2 = close.apply(lambda c: trend_r2(c, 60))

# ---- 8. Downside vol 60d (semi-deviation) ----
neg = ret.where(ret < 0, 0.0)
f_dvol = np.sqrt((neg ** 2).rolling(60).mean() * 252)

cands = {
    "dxy_beta_60d": f_dxy_beta,
    "risk_adj_mom_120d": f_ram,
    "dist_250d_high": f_d250,
    "trend_winrate_60d": f_win,
    "skew_60d": f_skew,
    "vix_beta_raw_60d": f_vixb,
    "trend_r2_60d": f_r2,
    "downside_vol_60d": f_dvol,
}

regime = "side/bearish 2026H2-2027, commodity complex downtrend, USD firm"
for name, f in cands.items():
    # standardize sign convention: report as-is; direction decided from IC sign
    run_validation(f, close, horizons=(1, 2, 3, 5, 10, 20),
                   factor_id=name, regime_notes=regime, return_summary=False)
    print()

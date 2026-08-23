"""/usr/bin/env python
miner_1 2033-01-06: Batch factor screening across the 15-asset cross-asset universe.
Visible through 2033-01-05 (persistent/date.json). Warm-up 2020-01-01..2026-07-15 is
research-window data; online data through 2033-01-05 is fully usable for validation.
Admission gates (shared): abs(IC)>=0.0070 and abs(ICIR)>=0.0840 at 10-day horizon.
"""
import sys, os, json, math
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_common import (WATCHLIST, load_panel, load_macro_panel, forward_returns,
                            spearman_ic_series, ic_metrics, coverage, turnover_rank_chg,
                            decay_by_horizon, max_library_corr, zlib_b64_panel, visible_through)

ADMISSION_IC = 0.0070
ADMISSION_ICIR = 0.0840
HORIZON = 10

panel, vpanel = load_panel("2020-01-01")
vt = visible_through()
print(f"visible_through={vt.date()} panel shape={panel.shape} volume cols={list(vpanel.columns)}")

spx = panel["SPX"]
r_spx = spx.pct_change()

def zscore(s):
    return (s - s.mean()) / (s.std() + 1e-12)

def rolling_beta(y, x, win):
    """Rolling beta of y on x over trailing win, aligned at last bar."""
    cov = y.rolling(win).cov(x)
    var = x.rolling(win).var()
    return cov / (var + 1e-12)

# Macro signals (observation-only)
vix_c = load_macro_panel("VIX")
dxy_c = load_macro_panel("DXY")
usdcny_c = load_macro_panel("USDCNY")
usdjpy_c = load_macro_panel("USDJPY")
eurusd_c = load_macro_panel("EURUSD")

r_vix = vix_c.pct_change()
vix_regime_high = (vix_c > vix_c.rolling(63).median()).astype(float)  # 1 if high-vol regime
vix_up = (vix_c > vix_c.shift(5)).astype(float)
dxy_ret = dxy_c.pct_change()

# ---------------- FACTOR DEFINITIONS ----------------
factors = {}

# F1: momentum skip5 (10/20/40)
for w, sk in [(10, 5), (20, 5), (40, 5)]:
    factors[f"mom_{w}d_skip{sk}"] = panel.shift(sk) / panel.shift(sk + w) - 1.0

# F2: high/low range position 20 (short-term reversal in range)
roll_max = panel.rolling(20).max()
roll_min = panel.rolling(20).min()
factors["range_pos_20"] = (panel - roll_min) / (roll_max - roll_min + 1e-12)

# F3: BB %B 20 (mean reversion)
sma20 = panel.rolling(20).mean()
sd20 = panel.rolling(20).std()
factors["bb_pctb_20"] = (panel - sma20) / (2 * sd20 + 1e-12)

# F4: vol of vol ratio 20x60 (relative vol regime)
vol20 = panel.pct_change().rolling(20).std()
vol60 = panel.pct_change().rolling(60).std()
factors["vol_of_vol20x60"] = vol20 / (vol60 + 1e-12)

# F5: downside ratio: rel downside vol in last 20 vs 60 (drawdown pressure)
down20 = panel.pct_change().clip(upper=0).rolling(20).std()
down60 = panel.pct_change().clip(upper=0).rolling(60).std()
factors["downside_ratio_20x60"] = down20 / (down60 + 1e-12)

# F6: up/down ratio 20 (asymmetry)
r = panel.pct_change()
up20 = r.clip(lower=0).rolling(20).mean()
down20b = (-r.clip(upper=0)).rolling(20).mean()
factors["up_down_ratio_20"] = up20 / (down20b + 1e-12)

# F7: eff ratio 20 (trend efficiency)
path = r.abs().rolling(20).sum()
net = (panel / panel.shift(20) - 1.0).abs()
factors["eff_ratio_20"] = net / (path + 1e-12)

# F8: VIX-beta 60 raw
factors["vix_beta_60"] = rolling_beta(panel.pct_change(), r_vix, 60)

# F9: VIX-beta conditional 60x20 (recompute)
vix_chg20 = vix_c / vix_c.shift(20) - 1.0
factors["vix_beta_cond_60x20"] = rolling_beta(panel.pct_change(), r_vix, 60) * -vix_chg20

# F10: yield beta conditional 60x20 (US10Y returns)
us10y_r = panel["US10Y"].pct_change()
y_chg20 = panel["US10Y"] / panel["US10Y"].shift(20) - 1.0
factors["yield_beta_cond_60x20"] = rolling_beta(r, us10y_r, 60) * -y_chg20

# F11: DXY beta 60
dxy_ret = dxy_c.pct_change()
factors["dxy_beta_60"] = rolling_beta(r, dxy_ret, 60)

# F12: DXY beta conditional 60x20
dxy_chg20 = dxy_c / dxy_c.shift(20) - 1.0
factors["dxy_beta_cond_60x20"] = rolling_beta(r, dxy_ret, 60) * -dxy_chg20

# F13: cross-asset dispersion (mean abs cross-section z-score of 5d returns) - date-level factor
r5 = panel.pct_change(5)
disp = r5.sub(r5.mean(axis=1), axis=0).abs().mean(axis=1)
disp_asset = pd.DataFrame(np.tile(disp.values[:, None], (1, panel.shape[1])),
                          index=panel.index, columns=panel.columns)
factors["cs_dispersion_5"] = -disp_asset  # expect negative IC -> defensive when dispersion high

# F14: cross-asset breadth (fraction of assets with positive 20d return)
mom20 = panel / panel.shift(20) - 1.0
breadth = (mom20 > 0).sum(axis=1) / panel.shape[1]
breadth_asset = pd.DataFrame(np.tile(breadth.values[:, None], (1, panel.shape[1])),
                             index=panel.index, columns=panel.columns)
factors["cs_breadth_20"] = breadth_asset

# F15: crypto-beta 60 (BTC beta)
btc_r = panel["BTC"].pct_change()
factors["crypto_beta_60"] = rolling_beta(r, btc_r, 60)

# F16: gold-beta 60 (XAU beta)
xau_r = panel["XAU"].pct_change()
factors["gold_beta_60"] = rolling_beta(r, xau_r, 60)

# F17: oil-beta 60 (WTI beta)
wti_r = panel["WTI"].pct_change()
factors["oil_beta_60"] = rolling_beta(r, wti_r, 60)

# F18: equity-beta 60 (SPX beta)
factors["equity_beta_60"] = rolling_beta(r, r_spx, 60)

# F19: vol-of-vol z-score 60
factors["vol_z_60"] = zscore(vol20)

# F20: kaufman efficiency 60
path60 = r.abs().rolling(60).sum()
net60 = (panel / panel.shift(60) - 1.0).abs()
factors["kaufman_eff_60"] = net60 / (path60 + 1e-12)

# F21: CV ratio 20 (coefficient of variation incl sign)
cv20 = vol20 / (1 + r.rolling(20).mean().abs())
factors["cv_ratio_20"] = -cv20

# F22: streak 20 (mean
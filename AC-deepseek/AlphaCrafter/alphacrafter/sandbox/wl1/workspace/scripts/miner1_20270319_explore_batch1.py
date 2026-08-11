"""miner_1 2027-03-19 cycle: explore candidate factor families (batch 1).

Context (trader feedback): side/bearish regime persists into 2027, commodity
complex drag (COPPER/WTI/XAU overweight failures), momentum anchor helped but
top-pick failures. Prior cycles covered dxy/vix beta, skew, downside vol,
risk-adj momentum, drawdown depth, trend R2/winrate.

New angles this cycle:
 1) Overnight vs intraday return decomposition (OHLC data, classic anomaly)
 2) Close-location-in-range (buying pressure / order flow proxy)
 3) Kaufman efficiency ratio (trend quality, not yet tested)
 4) Parkinson (high-low) vol vs close-close vol ratio (vol estimator gap)
 5) Volume trend & volume-price confirmation (liquidity/participation)
 6) Time-series momentum term structure (mom60 vs mom120 spread)
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner1_20270319_lib import load_panel, load_macro, run_validation

close = load_panel()
panel = pd.read_pickle("scripts/panel_cache.pkl")
O = panel["open"].reindex(close.index).ffill()
H = panel["high"].reindex(close.index).ffill()
L = panel["low"].reindex(close.index).ffill()
V = panel["vol"].reindex(close.index).ffill()
ret = close.pct_change()

print(f"close panel: {close.shape}  {close.index[0].date()} -> {close.index[-1].date()}")
print(f"weekday rows: {len(close)}")

# ---- 1. Overnight return component (20d mean) ----
overnight = O / close.shift(1) - 1.0
f_ovn_20 = overnight.rolling(20).mean()

# ---- 2. Intraday return component (20d mean) ----
intraday = close / O - 1.0
f_id_20 = intraday.rolling(20).mean()

# ---- 3. Close location in daily range (20d mean), 0=low 1=high ----
rng = (H - L).replace(0, np.nan)
loc = (close - L) / rng
f_loc_20 = loc.rolling(20).mean()

# ---- 4. Kaufman efficiency ratio 20d ----
def eff_ratio(s, n=20):
    num = (s - s.shift(n)).abs()
    den = s.diff().abs().rolling(n).sum()
    return num / den
f_eff20 = close.apply(lambda c: eff_ratio(c, 20))

# ---- 5. Parkinson vol 20d / close-close vol 20d (vol estimator gap) ----
park = np.sqrt((np.log(H / L) ** 2).rolling(20).mean() / (4 * np.log(2)))
cc_vol = ret.rolling(20).std()
f_pk_cc = (park / cc_vol).reindex(close.index)

# ---- 6. Volume trend 20d/120d ----
f_vtrend = V.rolling(20).mean() / V.rolling(120).mean()

# ---- 7. Volume-price confirmation: vol trend x sign(20d return) ----
f_vpconf = f_vtrend * np.sign(close / close.shift(20) - 1.0)

# ---- 8. TS momentum term structure: mom60 - mom120 ----
f_tsterm = (close / close.shift(60) - 1.0) - (close / close.shift(120) - 1.0)

cands = {
    "overnight_ret_20d": f_ovn_20,
    "intraday_ret_20d": f_id_20,
    "close_loc_20d": f_loc_20,
    "eff_ratio_20d": f_eff20,
    "park_cc_vol_ratio_20d": f_pk_cc,
    "vol_trend_20d_120d": f_vtrend,
    "vol_price_confirm_20d": f_vpconf,
    "ts_mom_term_60_120": f_tsterm,
}

regime = "side/bearish 2026H2-2027H1, commodity drag, USD firm, low vol"
for name, f in cands.items():
    run_validation(f, close, horizons=(1, 2, 3, 5, 10, 20),
                   factor_id=name, regime_notes=regime, return_summary=False)
    print()

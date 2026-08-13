"""miner_2 2032-08-05: diagnose why gap_freq_60 and cn10y_beta_60 have NaN recent IC."""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import WATCHLIST, load_prices, load_index, factor_to_panel, VAL_END

np.seterr(all='ignore')

prices = load_prices(days=3400)
max_date = max(dd.index.max() for dd in prices.values())
print(f"prices last date: {max_date.date()}")

# ---- CN10Y freeze point ----
cn = prices['CN10Y']['close']
d = cn.diff()
frozen_at = d[d != 0].index.max()
print(f"\nCN10Y: last non-zero daily diff on {frozen_at.date() if not pd.isna(frozen_at) else 'NA'}; "
      f"last {len(cn) - cn.index.get_loc(d[d != 0].index.max()) if not pd.isna(frozen_at) else len(cn)} rows constant")

# ---- gap_freq_60 panel OOS validity ----
def f_gap_freq_60(df, s):
    g = (df['open'] / df['close'].shift(1) - 1.0).abs()
    thr = g.rolling(120).median() * 1.5
    return (g > thr).astype(float).rolling(60).mean()

panel = factor_to_panel(f_gap_freq_60, prices)
oos = panel[panel.index >= VAL_END + pd.Timedelta(days=1)]
valid_per_date = oos.notna().sum(axis=1)
print(f"\ngap_freq_60 OOS dates: {len(oos)}, dates with >=8 valid: {(valid_per_date >= 8).sum()}")
print("valid counts by year (OOS):")
vc = valid_per_date.groupby(valid_per_date.index.year).agg(['count', 'mean', 'max'])
print(vc)
# where does it break?
last_good = valid_per_date[valid_per_date >= 8].index.max()
print(f"last date with >=8 valid: {last_good.date() if not pd.isna(last_good) else 'NA'}")
# sample panel near the break
print("\ngap_freq sample panel at 2026-09-30:")
for s in WATCHLIST:
    v = panel.loc['2026-09-30', s] if '2026-09-30' in panel.index else np.nan
    print(f"  {s:10s} {v}")

# ---- cn10y_beta_60 OOS validity ----
cn10y_d = prices['CN10Y']['close'].diff()
def f_cn10y_beta_60(df, s):
    z = pd.concat([df['close'].pct_change().rename('r'), cn10y_d.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['m']) / z['m'].rolling(60, min_periods=30).var().replace(0, np.nan)
    return b.reindex(df.index)
p2 = factor_to_panel(f_cn10y_beta_60, prices)
oos2 = p2[p2.index >= VAL_END + pd.Timedelta(days=1)]
vp2 = oos2.notna().sum(axis=1)
print(f"\ncn10y_beta_60 OOS dates: {len(oos2)}, dates with >=8 valid: {(vp2 >= 8).sum()}")
last_good2 = vp2[vp2 >= 8].index.max()
print(f"last date with >=8 valid: {last_good2.date() if not pd.isna(last_good2) else 'NA'}")

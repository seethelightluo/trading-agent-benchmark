"""miner_2 screening: explore multiple NEW factor ideas on 15-asset cross-section.
Uses data visible through 2033-11-25 only. One-off exploration; promising ideas
get dedicated validation scripts afterwards.
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from miner2_20331128_harness import (load_prices, load_macro, forward_returns,
                                     summarize, WATCH)

px = load_prices()
macro = load_macro()
fr = forward_returns(px)
ret = px.pct_change()

# ---------- candidate factor constructions ----------
factors = {}

# 1. skew_60d: skewness of daily returns over 60d
factors['skew_60d'] = ret.rolling(60).skew()

# 2. range_pos_20d: (close - min(low)) / (max(high)-min(low)) over 20d (uses close vs range)
high = px.rolling(20).max()
low = px.rolling(20).min()
factors['range_pos_20d'] = (px - low) / (high - low).replace(0, np.nan)

# 3. beta to XAU (gold) over 60d
r_xau = ret['XAU']
beta_xau = ret.rolling(60).cov(r_xau) / r_xau.rolling(60).var()
factors['beta_xau_60d'] = beta_xau

# 4. beta to WTI over 60d
r_wti = ret['WTI']
factors['beta_wti_60d'] = ret.rolling(60).cov(r_wti) / r_wti.rolling(60).var()

# 5. beta to BTC over 60d
r_btc = ret['BTC']
factors['beta_btc_60d'] = ret.rolling(60).cov(r_btc) / r_btc.rolling(60).var()

# 6. seasonality_month: avg same-calendar-month return over prior years (exclude last 30d)
def seasonality(px, lookback_years=5):
    r = px.pct_change()
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for col in r.columns:
        s = r[col]
        for dt in s.index:
            m = dt.month
            hist = s[(s.index.month == m) & (s.index < dt - pd.Timedelta(days=30))]
            if len(hist) >= 3:
                out.loc[dt, col] = hist.mean()
    return out
# vectorized-ish seasonal: use year-over-year same month mean
def seasonality_fast(px, max_years=6):
    r = px.pct_change()
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for col in r.columns:
        s = r[col]
        for m in range(1, 13):
            idx_m = s[s.index.month == m].index
            for dt in idx_m:
                hist = s[(s.index.month == m) & (s.index < dt) & (s.index >= dt - pd.DateOffset(years=max_years))]
                if len(hist) >= 3:
                    out.loc[dt, col] = hist.mean()
    return out
factors['seasonality_month'] = seasonality_fast(px)

# 7. amihud_20d: mean(|ret|)/mean(volume) over 20d (log-scaled)
vol = pd.concat([load_series(s)['volume'].rename(s) for s in WATCH], axis=1)
amihud = (ret.abs() / vol.replace(0, np.nan)).rolling(20).mean()
factors['amihud_20d'] = np.log1p(amihud * 1e6)

# 8. vol_adj_reversal_5d: negative short-term return scaled by 20d vol (reversal)
rv = ret.rolling(5).sum()
vol20 = ret.rolling(20).std()
factors['rev5_voladj'] = -rv / vol20

# 9. trend_persistence_60d (Hurst-like via variance ratio): std(60d ret)/ (sqrt(3)*std(20d ret)) ~ persistence
r60 = ret.rolling(60).sum()
r20 = ret.rolling(20).sum()
factors['vr_ratio_60_20'] = r60.rolling(60).std() / (np.sqrt(3) * r20.rolling(60).std())

# 10. max_gain_60d: max daily return over 60d (lottery preference)
factors['max_gain_60d'] = ret.rolling(60).max()

# 11. downside_beta_wti: beta in WTI-down days only (conditional)
mask_down = (r_wti < 0)
cov_d = ret[mask_down].rolling(60).cov(r_wti[mask_down]) / r_wti[mask_down].rolling(60).var()
factors['dn_beta_wti_60d'] = cov_d

# 12. gold_ratio_momentum: asset vs XAU relative 20d return (safe-haven rotation)
rel_xau = px / px['XAU']
factors['rel_mom_xau_20d'] = rel_xau.pct_change(20)

print(f'Data: {px.shape[0]} dates, {px.shape[1]} instruments, through {px.index[-1].date()}')
print()
for name, f in factors.items():
    try:
        summarize(f, fr, name)
    except Exception as e:
        print(f'{name}: ERROR {e}')

"""miner3 2031-05-09: SCREEN candidate factor families on cross-asset panel (exploration)."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr

panel = pd.read_pickle('scripts/panel_cache_20310509.pkl')
close = panel['close']
high = panel['high']
low = panel['low']
open_ = panel['open']
ret = close.pct_change()
macro = panel['macro']

MIN_VALID = 8
START = '2021-01-01'

def daily_ic(factor, fwd_ret):
    dates, ics = [], []
    for dt in factor.index:
        f = factor.loc[dt]
        r = fwd_ret.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < MIN_VALID:
            continue
        ic = spearmanr(f[mask], r[mask]).correlation
        if np.isnan(ic):
            continue
        dates.append(dt); ics.append(ic)
    ics = np.array(ics)
    if len(ics) == 0:
        return None
    return {'n': len(ics), 'ic': ics.mean(), 'icir': ics.mean()/ics.std() if ics.std()>0 else 0,
            'hit': (np.sign(ics)==np.sign(ics.mean())).mean()}

def evaluate(name, factor, horizons=(1,2,3,5,10)):
    factor = factor.reindex(close.index)
    out = []
    for h in horizons:
        fwd = close.shift(-h) / close - 1.0
        res = daily_ic(factor, fwd)
        if res:
            out.append((h, round(res['ic'],4), round(res['icir'],4), res['n']))
    print(f"{name:45s}", out)

# ---------------- candidate factors (each ~ per-asset cross-sectional) ----------------
# 1. Kaufman efficiency ratio 20d: trend quality
er20 = (close - close.shift(20)).abs() / ret.abs().rolling(20).sum()
# 2. Efficiency ratio 60d
er60 = (close - close.shift(60)).abs() / ret.abs().rolling(60).sum()
# 3. Downside semi-vol 20d (std of negative returns)
neg = ret.where(ret < 0, 0.0)
semivol20 = neg.rolling(20).std()
# 4. Downside/total vol ratio 60d
semivol60 = neg.rolling(60).std()
totalvol60 = ret.rolling(60).std()
asym60 = semivol60 / totalvol60
# 5. Cross-asset beta 60d (to equal-weight mean return)
mean_ret = ret.mean(axis=1)
beta60 = ret.rolling(60).cov(mean_ret) / mean_ret.rolling(60).var()
# 6. Distance from 60d high (drawdown) - negative
dd60 = close / close.rolling(60).max() - 1.0
# 7. Upper wick ratio 20d mean
tr = high - low
upper_wick = (high - np.maximum(open_, close)) / tr.replace(0, np.nan)
upper_wick20 = upper_wick.rolling(20).mean()
# 8. Body-to-range ratio 20d mean
body = (close - open_).abs() / tr.replace(0, np.nan)
body20 = body.rolling(20).mean()
# 9. Return kurtosis 60d
kurt60 = ret.rolling(60).kurt()
# 10. 20d momentum conditioned on VIX level (low VIX = risk-on, momentum works?)
vix = macro['VIX']
vix_regime = (vix - vix.rolling(120).mean()) / vix.rolling(120).std()
mom20 = close / close.shift(20) - 1.0
mom20_lowvix = mom20.where(vix_regime < 0)   # momentum when VIX low
mom20_highvix = mom20.where(vix_regime >= 0) # momentum when VIX high
# 11. USDJPY-conditional: JPY strength -> Japan risk-on?  conditional on 20d usdjpy move
usdjpy_chg20 = macro['USDJPY'].pct_change(20)
mom20_jpyweak = mom20.where(usdjpy_chg20 > 0)
# 12. Return skewness 40d
skew40 = ret.rolling(40).skew()
# 13. Volatility trend: 20d vol / 60d vol (vol rising?)
vol20 = ret.rolling(20).std(); vol60 = ret.rolling(60).std()
voltrend = vol20 / vol60

for name, fac in [
    ('ER20_trend_quality', er20), ('ER60_trend_quality', er60),
    ('semivol20_downside', semivol20), ('asym60_downside_ratio', asym60),
    ('beta60_crossasset', beta60), ('dd60_from_high', dd60),
    ('upper_wick20', upper_wick20), ('body_ratio20', body20),
    ('kurt60_tail', kurt60), ('skew40', skew40),
    ('mom20_lowvix', mom20_lowvix), ('mom20_highvix', mom20_highvix),
    ('mom20_jpyweak', mom20_jpyweak), ('voltrend20_60', voltrend),
]:
    try:
        fac = fac[fac.index >= START]
        evaluate(name, fac)
    except Exception as e:
        print(name, 'ERR', e)

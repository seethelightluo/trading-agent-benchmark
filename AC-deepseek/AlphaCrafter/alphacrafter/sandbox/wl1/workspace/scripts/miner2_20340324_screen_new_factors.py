"""miner2 2034-03-24: screen new candidate factors on the 15-name cross-asset panel.
Regime: VIX 51.4 crisis spike, high dispersion 1.88%, negative 20d drift.
Candidates (each is a single idea; screening pass to shortlist):
 bbp20       - 20d range (Bollinger) position
 oskew20     - negative 20d realized skewness
 dist_high60 - distance from 60d high (drawdown)
 volz5x60    - z-score of 5d realized vol vs 60d vol distribution
 rsi2        - 2-period RSI (short-term overbought/oversold)
 on_rev1     - overnight gap reversal 1d
 id_rev1     - intraday reversal 1d
 trange20    - 20d high-low range / close (vol level)
 lottery5    - max daily return over 5d (lottery demand)
 adx14       - 14d ADX trend strength
"""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner2_val_lib import eval_factor, summarize, fwd_ret, daily_rank_ic

with open('scripts/panel_cache_20340324.pkl', 'rb') as f:
    P = pd.read_pickle(f)
close, high, low, open_ = P['close'], P['high'], P['low'], P['open']
ret = P['ret']

factors = {}

# F1 bbp20: (close - low20) / (high20 - low20) - 0.5
lo20 = low.rolling(20).min(); hi20 = high.rolling(20).max()
factors['bbp20'] = (close - lo20) / (hi20 - lo20) - 0.5

# F2 oskew20: -skewness of daily returns over 20d
skew20 = ret.rolling(20).skew()
factors['oskew20'] = -skew20

# F3 dist_high60: close/rolling_max(high,60) - 1
factors['dist_high60'] = close / high.rolling(60).max() - 1.0

# F4 volz5x60: (vol5 - mean(vol5,60)) / std(vol5,60)
vol5 = ret.rolling(5).std()
mu = vol5.rolling(60).mean(); sd = vol5.rolling(60).std()
factors['volz5x60'] = (vol5 - mu) / sd

# F5 rsi2: 2-period RSI
def rsi(px, n=2):
    d = px.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn
    return 100 - 100 / (1 + rs)
factors['rsi2'] = rsi(close, 2)

# F6 on_rev1: -(open/prev_close - 1)
factors['on_rev1'] = -(open_ / close.shift(1) - 1.0)

# F7 id_rev1: -(close/open - 1)
factors['id_rev1'] = -(close / open_ - 1.0)

# F8 trange20: (high20 - low20)/close
factors['trange20'] = (hi20 - lo20) / close

# F9 lottery5: max daily return over 5d
factors['lottery5'] = ret.rolling(5).max()

# F10 adx14: 14d ADX (simplified Wilder)
def adx(px, n=14):
    up = px.diff(); dn = -px.diff()
    truerange = pd.concat([high - low, (high - px.shift(1)).abs(), (low - px.shift(1)).abs()], axis=1).max(axis=1)
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=px.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=px.index)
    atr = truerange.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1/n, adjust=False).mean()
factors['adx14'] = adx(close, 14)

print("=== SCREENING 2034-03-24 (data through 2034-03-23) ===")
full = {}
for name, sig in factors.items():
    res = eval_factor(sig, close, horizons=(1, 2, 3, 5, 10), min_n=8)
    h1 = summarize(res, label=name)
    full[name] = res

# recent 1y sub-window for timeliness
end = close.index.max()
start = end - pd.Timedelta(days=365)
print("\n=== RECENT 1Y (timeliness) ===")
for name, sig in factors.items():
    res = eval_factor(sig, close, horizons=(1, 2, 5), min_n=8, start=start, end=end)
    summarize(res, label=name + '_1y')

with open('scripts/miner2_20340324_screen_results.json', 'w') as f:
    import json
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    json.dump(clean(full), f, indent=1)
print("\nsaved screen results")

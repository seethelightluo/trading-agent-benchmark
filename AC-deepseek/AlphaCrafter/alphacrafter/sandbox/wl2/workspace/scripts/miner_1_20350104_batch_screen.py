"""Batch exploration of candidate factor ideas (screen only; follow-up scripts validate winners)."""
import sys
sys.path.insert(0, 'scripts')
from miner_1_20350104_common import *
import numpy as np

close_df, ret_df = load_all()
fwd = forward_returns(close_df)
H = 10
fwdH = fwd[H]

cands = {}

# 1. RSI-14 (Wilder)
def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)
cands['rsi_14'] = rsi(close_df)

# 2. drawdown from 252d high (negative) and 60d high
cands['dd_252'] = close_df / close_df.rolling(252, min_periods=60).max() - 1.0
cands['dd_60'] = close_df / close_df.rolling(60, min_periods=30).max() - 1.0

# 3. vol-scaled momentum 20d / vol60
mom20 = close_df.pct_change(20)
vol60 = ret_df.rolling(60).std() * np.sqrt(252)
cands['volmom20_60'] = mom20 / vol60
cands['sharpe_60'] = close_df.pct_change(60) / vol60

# 4. Hurst exponent via R/S, 60d window
def hurst(series, window=60, min_period=40):
    out = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(window, len(series)):
        seg = vals[i-window:i]
        if np.any(~np.isfinite(seg)):
            continue
        mean_r = seg.mean()
        dev = seg - mean_r
        cum = np.cumsum(dev)
        R = cum.max() - cum.min()
        S = seg.std(ddof=1)
        if S <= 0 or R <= 0:
            continue
        out.iloc[i] = np.log(R / S) / np.log(window)
    return out
cands['hurst_60'] = hurst(ret_df)

# 5. skew 60d, kurtosis 60d
cands['skew_60'] = ret_df.rolling(60, min_periods=40).skew()
cands['cvar20'] = ret_df.rolling(20, min_periods=15).quantile(0.05)

# 6. return autocorrelation 5d (sign flip for reversal)
cands['autocorr_5'] = ret_df.rolling(10).apply(lambda x: np.corrcoef(x[:-5], x[5:])[0, 1] if np.std(x[:-5])>0 and np.std(x[5:])>0 else np.nan, raw=True)
# 7. close/MA50 - 1, close/MA100 - 1, BB zscore 60
cands['ma_dist_50'] = close_df / close_df.rolling(50, min_periods=30).mean() - 1.0
cands['ma_dist_100'] = close_df / close_df.rolling(100, min_periods=60).mean() - 1.0
bb_mean = close_df.rolling(60, min_periods=40).mean()
bb_std = close_df.rolling(60, min_periods=40).std()
cands['bb_zscore_60'] = (close_df - bb_mean) / bb_std

# 8. relative strength vs SPX 60d and VIX beta 60d
cands['rel_spx_60'] = close_df.pct_change(60) - close_df['SPX'].pct_change(60)
vix_chg = close_df['VIX'].pct_change()
cands['vix_beta_60'] = ret_df.rolling(60, min_periods=40).cov(vix_chg) / vix_chg.rolling(60, min_periods=40).var()

for name, f in cands.items():
    f = f.reindex(close_df.index)
    m = eval_factor(f, fwdH, label=name)
    print_metrics(m)
    print('  yearly:', yearly_ic(f, fwdH))

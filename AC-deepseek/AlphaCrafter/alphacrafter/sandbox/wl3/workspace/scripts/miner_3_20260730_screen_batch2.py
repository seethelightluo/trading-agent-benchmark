"""miner_3 2026-07-30 batch screen #2: trend-quality / tail-risk / vol-compression family.

Candidates (all full-coverage price-based except cn10y cond beta):
 1. autocorr_1_60   : lag-1 return autocorrelation over 60d (trend persistence).
 2. max_ret_20d     : max daily return over 20d (MAX/lottery effect).
 3. upday_ratio_60  : fraction of up days over 60d minus 0.5 (trend consistency).
 4. bw_zscore_20_60 : Bollinger bandwidth (2*std20/ma20) z-scored over 60d (vol compression).
 5. rsi_14          : classic RSI(14) (mean reversion).
 6. gk_vol_ratio_20 : Garman-Klass range vol / close-close vol over 20d (intraday info ratio).
 7. skew_term_20_60 : skew(20) - skew(60) (skewness term structure).
 8. cn10y_beta_cond_60x20 : beta(asset, CN10Y chg, 60) * 20d CN10Y move (China rates driver).
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import load_prices, evaluate_candidate

prices = load_prices(days=2000)
cn10y = prices.get('CN10Y')
print(f"loaded {len(prices)} assets; CN10Y len={0 if cn10y is None else len(cn10y)}")


def autocorr_1_60(df, s):
    r = df['close'].pct_change()
    a = r.rolling(60).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 3 else np.nan, raw=False)
    return a


def max_ret_20d(df, s):
    return df['close'].pct_change().rolling(20).max()


def upday_ratio_60(df, s):
    r = df['close'].pct_change()
    up = (r > 0).astype(float).rolling(60).mean()
    return up - 0.5


def bw_zscore_20_60(df, s):
    c = df['close']
    ma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    bw = 2.0 * sd / ma
    mu = bw.rolling(60).mean()
    s = bw.rolling(60).std()
    return (bw - mu) / s.replace(0, np.nan)


def rsi_14(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0).rolling(14).mean()
    dn = (-r.clip(upper=0)).rolling(14).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def gk_vol_ratio_20(df, s):
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    log_h = np.log(h / o)
    log_l = np.log(l / o)
    log_c = np.log(c / o)
    gk = 0.5 * log_h ** 2 - (2 * np.log(2) - 1) * log_c ** 2 + log_h * log_l
    gk = np.sqrt(gk.clip(lower=0))
    cc = df['close'].pct_change().rolling(20).std()
    return (gk.rolling(20).mean() / cc.replace(0, np.nan))


def skew_term_20_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()


def cn10y_beta_cond_60x20(df, s):
    if cn10y is None:
        return None
    r = df['close'].pct_change()
    rc = cn10y['close'].pct_change()
    z = pd.concat([r.rename('r'), rc.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['c']) / z['c'].rolling(60).var()
    c_move = cn10y['close'] / cn10y['close'].shift(20) - 1.0
    return (b * c_move).reindex(z.index)


cands = [
    ('autocorr_1_60', autocorr_1_60),
    ('max_ret_20d', max_ret_20d),
    ('upday_ratio_60', upday_ratio_60),
    ('bw_zscore_20_60', bw_zscore_20_60),
    ('rsi_14', rsi_14),
    ('gk_vol_ratio_20', gk_vol_ratio_20),
    ('skew_term_20_60', skew_term_20_60),
    ('cn10y_beta_cond_60x20', cn10y_beta_cond_60x20),
]

for fid, fn in cands:
    print('#' * 72)
    m, panel = evaluate_candidate(fid, fn, prices)
    if m is not None:
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"SCREEN: {fid:22s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"cov={m['coverage_asset_days']:.3f} turn={m['turnover_10d_rank']:.2f} "
              f"rho={m.get('max_abs_library_correlation', float('nan')):.3f} -> {'PASS' if ok else 'FAIL'}")

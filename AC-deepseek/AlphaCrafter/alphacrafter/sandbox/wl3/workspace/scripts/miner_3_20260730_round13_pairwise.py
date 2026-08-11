"""Round 13 winners: pairwise candidate correlation (self-contained, no import side effects)."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'scripts')
from factor_common import load_prices, canonical_grid, factor_to_panel, signal_matrix

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
ret_panel = pd.DataFrame({s: prices[s]['close'].pct_change() for s in
                          ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
                           'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']})
mkt_ret = ret_panel.mean(axis=1)


def entropy_sign_20(df, s):
    pos = (df['close'].pct_change() > 0).astype(float)
    n_up = pos.rolling(20).sum()
    p = n_up / 20.0
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0)
    return (1.0 - ent).clip(0.0, 1.0)


def conviction_20(df, s):
    r = df['close'].pct_change()
    mom20 = df['close'] / df['close'].shift(20) - 1.0
    vol20 = r.rolling(20).std() * np.sqrt(20)
    return (mom20 / vol20.replace(0, np.nan)).abs()


def gap_freq_20(df, s):
    pc = df['close'].shift(1)
    gap = df['open'] / pc - 1.0
    tr = pd.concat([(df['high'] - df['low']),
                    (df['high'] - pc).abs(),
                    (df['low'] - pc).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    big = (gap.abs() > 0.5 * atr14).astype(float)
    return big.rolling(20).mean()


def downside_dev_ratio_20_60(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0.0)
    dd20 = (neg ** 2).rolling(20).mean().apply(np.sqrt)
    dd60 = (neg ** 2).rolling(60).mean().apply(np.sqrt)
    return dd20 / dd60.replace(0, np.nan)


winners = {
    'entropy_sign_20': entropy_sign_20,
    'conviction_20': conviction_20,
    'gap_freq_20': gap_freq_20,
    'downside_dev_ratio_20_60': downside_dev_ratio_20_60,
}
mats = {fid: signal_matrix(factor_to_panel(fn, prices), grid) for fid, fn in winners.items()}
ids = list(mats)
print("pairwise mean daily cross-sectional Spearman rho (canonical grid):")
for i, a in enumerate(ids):
    for j, b in enumerate(ids):
        if j <= i:
            continue
        corrs = []
        xa, xb = mats[a], mats[b]
        for d in range(xa.shape[0]):
            x, y = xa[d], xb[d]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                r = np.corrcoef(np.argsort(np.argsort(x[m])), np.argsort(np.argsort(y[m])))[0, 1]
                if np.isfinite(r):
                    corrs.append(r)
        print(f"  {a:26s} vs {b:26s}: {np.mean(corrs):+.3f}")

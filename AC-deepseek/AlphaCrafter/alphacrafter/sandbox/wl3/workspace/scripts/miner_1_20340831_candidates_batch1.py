"""miner_1 2034-08-31 candidate factor screen (batch 1).

Validates one idea per candidate across:
  WARM   : 2020-01-01..2026-07-15 (canonical admission reference, all 15 assets)
  OOS    : 2026-07-16..last (live assets only)
  RECENT : last ~365d (live assets only)
Admission gate (shared): |IC|>=0.007 and |ICIR|>=0.084 at h=10 on WARM.
Library correlation: max abs mean daily cross-sectional Spearman vs persisted
signal artifacts (factors/*_signal.npy on the canonical grid).
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, forward_returns, rank_ic_series,
                           VAL_START, VAL_END, canonical_grid, signal_matrix)
from miner1_eval_helper import eval_candidate, stats, load_library_artifacts, library_corr_matrix

# ---------- load data with full history ----------
prices = load_prices(days=5000)
print('assets loaded:', {s: len(d) for s, d in prices.items()})
max_date = max(dd.index.max() for dd in prices.values())
print('max visible date:', max_date.date())

# ---------- verify canonical grid matches library artifacts ----------
grid = canonical_grid(prices)
lib_mats = load_library_artifacts()
print('canonical grid n_dates:', len(grid), grid.min().date(), '..', grid.max().date())
for fid, m in list(lib_mats.items())[:3]:
    print('  artifact', fid, m.shape)

# ---------- candidate factor definitions ----------
def f_autocorr20(df, s):
    r = df['close'].pct_change()
    def ac(w):
        if len(w) < 8:
            return np.nan
        x, y = w[:-1], w[1:]
        if np.std(x) == 0 or np.std(y) == 0:
            return np.nan
        return np.corrcoef(x, y)[0, 1]
    return r.rolling(20).apply(ac, raw=True)

def f_overnight_share_20(df, s):
    prev_close = df['close'].shift(1)
    on = df['open'] / prev_close - 1.0
    intra = df['close'] / df['open'] - 1.0
    num = on.rolling(20).mean()
    den = (on.abs() + intra.abs()).rolling(20).mean()
    return num / den.replace(0, np.nan)

def f_tail_kurt_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60).kurt()

def f_obv_mom_20(df, s):
    r = df['close'].pct_change()
    v = df['volume']
    obv = (np.sign(r) * v).cumsum()
    # 20d net volume flow as fraction of 20d total volume
    flow = obv.diff(20)
    tot = v.rolling(20).sum()
    return flow / tot.replace(0, np.nan)

def f_downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0)
    ds = (neg ** 2).rolling(60).mean().apply(np.sqrt)
    tot = r.rolling(60).std()
    return ds / tot.replace(0, np.nan)

def f_close_pos_mean_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    pos = (df['close'] - df['low']) / rng
    return pos.rolling(20).mean()

def f_vol_trend_20_120(df, s):
    v = df['volume']
    return v.rolling(20).mean() / v.rolling(120).mean().replace(0, np.nan)

def f_vol_ratio_5_120(df, s):
    r = df['close'].pct_change()
    return r.rolling(5).std() / r.rolling(120).std().replace(0, np.nan)

def make_cs_mom_z_20(prices):
    """Cross-sectional z-score of 20d momentum (skip5) across the 15-asset basket."""
    cols = {}
    for s, df in prices.items():
        cols[s] = df['close'].shift(5) / df['close'].shift(25) - 1.0
    panel = pd.DataFrame(cols).sort_index()
    mu = panel.mean(axis=1)
    sd = panel.std(axis=1)
    z = panel.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
    return z

CANDIDATES = [
    ('ret_autocorr_20', lambda p: _panel(p, f_autocorr20)),
    ('overnight_share_20', lambda p: _panel(p, f_overnight_share_20)),
    ('tail_kurt_60', lambda p: _panel(p, f_tail_kurt_60)),
    ('obv_mom_20', lambda p: _panel(p, f_obv_mom_20)),
    ('downside_vol_ratio_60', lambda p: _panel(p, f_downside_vol_ratio_60)),
    ('close_pos_mean_20', lambda p: _panel(p, f_close_pos_mean_20)),
    ('vol_trend_20_120', lambda p: _panel(p, f_vol_trend_20_120)),
    ('vol_ratio_5_120', lambda p: _panel(p, f_vol_ratio_5_120)),
    ('cs_mom_z_20', make_cs_mom_z_20),
]

def _panel(prices, fn):
    cols = {}
    for s, df in prices.items():
        try:
            ser = fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception as e:
            print('  ERR', s, e)
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols)
    panel = panel[~panel.index.duplicated(keep='last')].sort_index()
    return panel

results = []
for fid, panel_fn in CANDIDATES:
    try:
        res = eval_candidate(fid, panel_fn, prices=prices, print_out=True)
        results.append(res)
    except Exception as e:
        print(f'{fid} FAILED: {e}', flush=True)

with open('scripts/miner_1_20340831_results_batch1.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('saved results to scripts/miner_1_20340831_results_batch1.json')

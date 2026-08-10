"""miner_2 2026-07-30 batch screen: candidate factor ideas (one idea per fn).

Run a slice of candidates:  python miner_2_20260730_screen_batch.py <start> <end>
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import (load_prices, load_index, evaluate_candidate,
                           build_library_panels)

prices = load_prices(days=2000)
dxy = load_index('DXY')
lib = build_library_panels(prices)

def f_rsi_14(df, s):
    close = df['close']
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).replace([np.inf], np.nan)

def f_bollinger_z_20(df, s):
    close = df['close']
    sma = close.rolling(20).mean()
    sd = close.rolling(20).std()
    return (close - sma) / sd.replace(0, np.nan)

def f_updown_ratio_60(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0).rolling(60).mean()
    dn = (-r.where(r < 0)).rolling(60).mean()
    return (up / dn.replace(0, np.nan)).replace([np.inf], np.nan)

def f_mom_ratio_120_20(df, s):
    close = df['close']
    m120 = close.shift(5) / close.shift(125) - 1.0
    m20 = close.shift(5) / close.shift(25) - 1.0
    return m120 / m20.abs().replace(0, np.nan)

def f_eff_ratio_60(df, s):
    close = df['close']
    num = (close - close.shift(60)).abs()
    den = close.pct_change().abs().rolling(60).sum()
    return num / den.replace(0, np.nan)

def f_downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    neg = r.clip(upper=0)
    return neg.rolling(60).std() / r.rolling(60).std().replace(0, np.nan)

def f_skew_60(df, s):
    return df['close'].pct_change().rolling(60).skew()

def f_dxy_beta_cond_60x20(df, s):
    if dxy is None:
        return None
    r = df['close'].pct_change()
    dr = dxy['close'].pct_change()
    z = pd.concat([r.rename('r'), dr.rename('d')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['d']) / z['d'].rolling(60).var().replace(0, np.nan)
    dxy_move = dxy['close'] / dxy['close'].shift(20) - 1.0
    return (b * dxy_move).reindex(z.index)

def f_gw_high_252(df, s):
    roll_max = df['close'].rolling(252, min_periods=60).max()
    return df['close'] / roll_max - 1.0

def f_vol_adj_mom_20_60(df, s):
    close = df['close']
    mom = close.shift(5) / close.shift(25) - 1.0
    vol = close.pct_change().rolling(60).std()
    return mom / vol.replace(0, np.nan)

def f_vol_term_20_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).std() / r.rolling(60).std().replace(0, np.nan) - 1.0

def f_ret_zscore_20(df, s):
    close = df['close']
    r = close.pct_change()
    mu = r.rolling(20).mean()
    sd = r.rolling(20).std()
    return ((r - mu) / sd.replace(0, np.nan)).shift(1)

def f_vol_ratio_20_120(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).std() / r.rolling(120).std().replace(0, np.nan) - 1.0

def f_high_low_range_pos_20(df, s):
    close = df['close']
    hi = df['high'].rolling(20).max()
    lo = df['low'].rolling(20).min()
    return (close - lo) / (hi - lo).replace(0, np.nan)

CANDIDATES = [
    ("rsi_14d", f_rsi_14),
    ("bollinger_z_20d", f_bollinger_z_20),
    ("updown_ratio_60d", f_updown_ratio_60),
    ("mom_ratio_120_20", f_mom_ratio_120_20),
    ("eff_ratio_60d", f_eff_ratio_60),
    ("downside_vol_ratio_60x20", f_downside_vol_ratio_60),
    ("skew_60d", f_skew_60),
    ("dxy_beta_cond_60x20", f_dxy_beta_cond_60x20),
    ("gw_high_252", f_gw_high_252),
    ("vol_adj_mom_20_60", f_vol_adj_mom_20_60),
    ("vol_term_20_60", f_vol_term_20_60),
    ("ret_zscore_20d", f_ret_zscore_20),
    ("vol_ratio_20_120", f_vol_ratio_20_120),
    ("high_low_range_pos_20", f_high_low_range_pos_20),
]

if __name__ == '__main__':
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end = int(sys.argv[2]) if len(sys.argv) > 2 else len(CANDIDATES)
    slice_ = CANDIDATES[start:end]
    print(f"slice [{start}:{end}] of {len(CANDIDATES)} candidates")
    for fid, fn in slice_:
        try:
            m, panel = evaluate_candidate(fid, fn, prices, library_panels=lib, print_out=False)
        except Exception as exc:
            print(f"{fid:28s} ERROR {exc}")
            continue
        if m is None:
            print(f"{fid:28s} INSUFFICIENT")
            continue
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"{fid:28s} IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']:5d} cov={m['coverage_asset_days']:.2f} turn={m['turnover_10d_rank']:.2f} "
              f"rho_lib={m['max_abs_library_correlation']:.2f} vs {m['max_corr_library_id']} -> {'PASS' if ok else 'FAIL'}")
        d = m['decay_ic_by_horizon']
        print(f"{'':28s} decay " + " ".join(f"h{h}:{d[str(h)]:+.4f}" for h in [1, 3, 5, 10, 20]))

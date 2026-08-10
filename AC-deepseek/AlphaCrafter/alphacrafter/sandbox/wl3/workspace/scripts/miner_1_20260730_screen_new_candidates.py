"""miner_1 2026-07-30: screen NEW factor candidates (batch 1).

All candidates are new ideas not previously persisted in factors/.
Full validation battery via factor_common (IC/ICIR at 10d horizon, coverage,
turnover, decay, max_abs_library_correlation).
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, evaluate_candidate,
                           build_library_panels, VAL_START, VAL_END)

prices = load_prices(days=2200)
print('loaded assets:', len(prices))
print('date range:', min(df.index.min() for df in prices.values()).date(),
      '..', max(df.index.max() for df in prices.values()).date())

# observation-only macro signals
dxy = load_index('DXY', prices=prices)
usdjpy = load_index('USDJPY', prices=prices)
eurusd = load_index('EURUSD', prices=prices)
vix = load_index('VIX', prices=prices)
lib = build_library_panels(prices)
print('library panels:', {k: v.shape for k, v in lib.items()})

CANDIDATES = {}

# S1: Close location value 20d (intraday range position, averaged)
def f_clv_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    clv = (df['close'] - df['low']) / rng
    return clv.rolling(20).mean()
CANDIDATES['clv_20'] = f_clv_20

# S2: Downside momentum 20d: average negative daily return (asymmetry)
def f_down_mom_20(df, s):
    r = df['close'].pct_change()
    return r.clip(upper=0).rolling(20).mean()
CANDIDATES['down_mom_20'] = f_down_mom_20

# S3: Vol expansion 20x60: vol20 / vol60 - 1
def f_vol_exp_20x60(df, s):
    v20 = df['close'].pct_change().rolling(20).std()
    v60 = df['close'].pct_change().rolling(60).std()
    return (v20 / v60.replace(0, np.nan)) - 1.0
CANDIDATES['vol_exp_20x60'] = f_vol_exp_20x60

# S4: Trend steepness: close/SMA20 - close/SMA60
def f_trend_steep(df, s):
    c = df['close']
    return c / c.rolling(20).mean() - c / c.rolling(60).mean()
CANDIDATES['trend_steep'] = f_trend_steep

# S5: Stochastic %K 20d position
def f_stoch_k_20(df, s):
    lo = df['close'].rolling(20).min()
    hi = df['close'].rolling(20).max()
    return (df['close'] - lo) / (hi - lo).replace(0, np.nan)
CANDIDATES['stoch_k_20'] = f_stoch_k_20

# S6: Aroon oscillator 25d
def f_aroon_25(df, s):
    n = 25
    hi = df['high'].rolling(n, min_periods=n).apply(lambda x: n - 1 - x.values.argmax(), raw=True)
    lo = df['low'].rolling(n, min_periods=n).apply(lambda x: n - 1 - x.values.argmin(), raw=True)
    return (hi - lo) / n
CANDIDATES['aroon_25'] = f_aroon_25

# S7: USDJPY-beta conditional: asset beta to USDJPY * USDJPY 20d move
def f_usdjpy_beta_cond(df, s):
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    jr = usdjpy['close'].pct_change()
    z = pd.concat([r.rename('r'), jr.rename('j')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['j']) / z['j'].rolling(60).var().replace(0, np.nan)
    move = usdjpy['close'] / usdjpy['close'].shift(20) - 1.0
    return (b * move).reindex(z.index)
CANDIDATES['usdjpy_beta_cond_60x20'] = f_usdjpy_beta_cond

# S8: EURUSD-beta conditional
def f_eurusd_beta_cond(df, s):
    if eurusd is None:
        return None
    r = df['close'].pct_change()
    er = eurusd['close'].pct_change()
    z = pd.concat([r.rename('r'), er.rename('e')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['e']) / z['e'].rolling(60).var().replace(0, np.nan)
    move = eurusd['close'] / eurusd['close'].shift(20) - 1.0
    return (b * move).reindex(z.index)
CANDIDATES['eurusd_beta_cond_60x20'] = f_eurusd_beta_cond

# S9: Momentum acceleration: mom20_skip5 - mom60_skip5
def f_mom_accel(df, s):
    c = df['close']
    return (c.shift(5) / c.shift(25) - 1.0) - (c.shift(5) / c.shift(65) - 1.0)
CANDIDATES['mom_accel_20_60'] = f_mom_accel

# S10: Keltner channel position 20d (ATR-normalized)
def f_keltner_pos_20(df, s):
    c = df['close']
    sma = c.rolling(20).mean()
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - c.shift(1)).abs(),
                    (df['low'] - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20).mean().replace(0, np.nan)
    return (c - (sma - 2 * atr)) / (4 * atr)
CANDIDATES['keltner_pos_20'] = f_keltner_pos_20

# S11: Signed skewness 20d (skew * sign of 20d return)
def f_skew20_signed(df, s):
    r = df['close'].pct_change()
    sk = r.rolling(20).skew()
    m = df['close'] / df['close'].shift(20) - 1.0
    return sk * np.sign(m)
CANDIDATES['skew20_signed'] = f_skew20_signed

# S12: Overnight/intraday return mix 20d: avg overnight - avg intraday
def f_ni_split_20(df, s):
    over = df['open'] / df['close'].shift(1) - 1.0
    intra = df['close'] / df['open'] - 1.0
    return (over - intra).rolling(20).mean()
CANDIDATES['ni_split_20'] = f_ni_split_20

# S13: VIX-beta conditional (VIX level change * beta) -- variant without sign flip
def f_vixbeta_level_60x20(df, s):
    if vix is None:
        return None
    r = df['close'].pct_change()
    vr = vix['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var().replace(0, np.nan)
    move = vix['close'] / vix['close'].shift(20) - 1.0
    return (b * move).reindex(z.index)
CANDIDATES['vixbeta_level_60x20'] = f_vixbeta_level_60x20

# S14: Upper wick 10d (short-term supply pressure)
def f_upper_wick_10(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    uw = (df['high'] - np.maximum(df['open'], df['close'])) / rng
    return uw.rolling(10).mean()
CANDIDATES['upper_wick_10'] = f_upper_wick_10

# S15: Rolling 60d return autocorrelation lag-2
def f_autocorr2_60(df, s):
    r = df['close'].pct_change()
    def ac2(x):
        if len(x) < 5:
            return np.nan
        xx = pd.Series(x)
        return xx.autocorr(lag=2)
    return r.rolling(60).apply(ac2, raw=False)
CANDIDATES['autocorr2_60'] = f_autocorr2_60

# S16: Distance below 20d high, vol-scaled (pullback depth)
def f_pullback_20(df, s):
    c = df['close']
    hi = c.rolling(20).max()
    v = c.pct_change().rolling(20).std().replace(0, np.nan)
    return (c / hi - 1.0) / v
CANDIDATES['pullback_20_vol'] = f_pullback_20

results = []
for fid, fn in CANDIDATES.items():
    try:
        m, panel = evaluate_candidate(fid, fn, prices, library_panels=lib)
        if m is None:
            results.append((fid, None, None, None))
            continue
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        corr_ok = m.get('max_abs_library_correlation', 1.0) is not None and m.get('max_abs_library_correlation', 1.0) < 0.5
        results.append((fid, m['ic'], m['icir'], m.get('max_abs_library_correlation')))
        print(f'RESULT {fid:26s} IC={m["ic"]:+.4f} ICIR={m["icir"]:+.4f} '
              f'hit={m["ic_hit_ratio"]:.3f} cov={m["coverage_asset_days"]:.2f} '
              f'turn={m["turnover_10d_rank"]:.2f} rho={m.get("max_abs_library_correlation")} '
              f'({m.get("max_corr_library_id")}) ndates={m["n_ic_dates"]}')
        print(f'  ADMISSION: |IC|>={0.007} {abs(m["ic"])>=0.007} | |ICIR|>={0.084} {abs(m["icir"])>=0.084} | rho<0.5 {corr_ok} -> {"PASS" if ok and corr_ok else "FAIL"}')
        print('  decay:', json.dumps(m['decay_ic_by_horizon'], default=str))
    except Exception as e:
        print(f'ERROR {fid}: {e}')
        results.append((fid, None, None, None))

print('\n=== SUMMARY ===')
for fid, ic, icir, rho in results:
    if ic is not None:
        flag = 'PASS' if (abs(ic) >= 0.007 and abs(icir) >= 0.084 and rho is not None and rho < 0.5) else 'fail'
        print(f'{fid:26s} IC={ic:+.4f} ICIR={icir:+.4f} rho={rho} -> {flag}')
    else:
        print(f'{fid:26s} no metrics')

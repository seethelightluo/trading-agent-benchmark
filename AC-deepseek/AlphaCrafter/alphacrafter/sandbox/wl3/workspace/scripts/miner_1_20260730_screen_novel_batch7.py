"""miner_1 cycle 2026-07-30: screen novel factor candidates (batch 7).

All candidates are distinct from the 12-factor effective library and prior
rejected/evicted/quarantined sets. Each candidate is validated on the shared
2020-01-01..2026-07-15 warm-up window at h=10 with the admission gates
|IC| >= 0.0070, |ICIR| >= 0.0840.
"""
import sys
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from factor_common import load_prices, factor_to_panel, validate_factor, VAL_START, VAL_END

prices = load_prices(days=2000)
print(f"prices loaded: {len(prices)} assets, window {min(d.index.min() for d in prices.values()).date()}..{max(d.index.max() for d in prices.values()).date()}")


def f_parkinson_vol_20(df, s):
    # Parkinson range vol: sqrt(mean(ln(H/L)^2)/(4 ln2)) over 20d, negative -> low range vol
    hl = np.log(df['high'] / df['low'])
    v = (hl.rolling(20).apply(lambda x: np.sqrt(np.nanmean(x ** 2) / (4 * np.log(2))), raw=True))
    return -v


def f_ma_slope_20_60(df, s):
    # normalized 20d SMA slope over 60d: (sma20/sma20.shift(60)-1) / vol_20
    sma = df['close'].rolling(20).mean()
    mom = sma / sma.shift(60) - 1.0
    vol = df['close'].pct_change().rolling(20).std()
    return (mom / (vol * np.sqrt(60)))


def f_amihud_20(df, s):
    # Amihud illiquidity: mean(|ret|/volume) * 1e9, negative (illiquid -> lower)
    r = df['close'].pct_change().abs()
    illiq = (r / df['volume'].replace(0, np.nan)).rolling(20).mean() * 1e9
    return -illiq


def f_max_dd_60(df, s):
    # max drawdown depth over 60d (negative depth)
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    return dd.rolling(60, min_periods=20).min()


def f_kurt_20(df, s):
    # rolling kurtosis of 20d returns
    return df['close'].pct_change().rolling(20).kurt()


def f_gain_loss_asym_20(df, s):
    # mean positive ret / |mean negative ret| over 20d
    r = df['close'].pct_change()
    pos = r.clip(lower=0).rolling(20).mean()
    neg = r.clip(upper=0).rolling(20).mean().abs()
    return pos / neg.replace(0, np.nan)


def f_vol_ma_ratio_20_120(df, s):
    # vol regime: vol_20 / vol_120 (high current vol vs long-run)
    r = df['close'].pct_change()
    v20 = r.rolling(20).std()
    v120 = r.rolling(120).std()
    return v20 / v120


def f_beta_ndx_60(df, s):
    # beta of asset returns vs NDX over 60d
    ndx = prices['NDX']['close']
    r = df['close'].pct_change()
    rn = ndx.pct_change()
    z = pd.concat([r.rename('r'), rn.rename('n')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['n']) / z['n'].rolling(60).var()
    return b


def f_mom_accel_20_60(df, s):
    # momentum acceleration: (mom_20_skip5 - mom_60_skip5)
    c = df['close']
    m20 = c.shift(5) / c.shift(25) - 1.0
    m60 = c.shift(5) / c.shift(65) - 1.0
    return m20 - m60


def f_days_since_high_20(df, s):
    # negative days since 20d high (recently at highs -> high score)
    c = df['close']
    roll_max = c.rolling(20).max()
    since = (c / roll_max)  # closeness to high
    return since


def f_obv_slope_20(df, s):
    # OBV slope over 20d normalized by vol
    r = df['close'].pct_change()
    obv = (np.sign(r) * df['volume']).cumsum()
    slope = obv - obv.shift(20)
    vol = r.rolling(20).std()
    return slope / (df['volume'].rolling(20).mean() * vol.replace(0, np.nan) + 1e-9)


def f_inv_vol_20(df, s):
    # inverse realized vol (risk-parity tilt)
    r = df['close'].pct_change()
    v = r.rolling(20).std()
    return 1.0 / v


CANDIDATES = [
    ("parkinson_vol_20", f_parkinson_vol_20),
    ("ma_slope_20_60", f_ma_slope_20_60),
    ("amihud_20", f_amihud_20),
    ("max_dd_60", f_max_dd_60),
    ("kurt_20", f_kurt_20),
    ("gain_loss_asym_20", f_gain_loss_asym_20),
    ("vol_ma_ratio_20_120", f_vol_ma_ratio_20_120),
    ("beta_ndx_60", f_beta_ndx_60),
    ("mom_accel_20_60", f_mom_accel_20_60),
    ("days_since_high_20", f_days_since_high_20),
    ("obv_slope_20", f_obv_slope_20),
    ("inv_vol_20", f_inv_vol_20),
]

for fid, fn in CANDIDATES:
    try:
        panel = factor_to_panel(fn, prices)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f"{fid}: NO VALID IC -> None")
            continue
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        print(f"{fid}: IC={m['ic']:+.4f} ICIR={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} decay10={m['decay_ic_by_horizon']['10']:+.4f} "
              f"decay20={m['decay_ic_by_horizon']['20']:+.4f} -> {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"{fid}: ERROR {e}")

"""miner_1 2035-05-10: explore fresh factor candidates on 15-instrument universe.
Visible-through 2035-05-09. No lookahead: signals use data <= VISIBLE_END,
forward returns measured after. Admission gate: |IC|>=0.0070 and |ICIR|>=0.084 at h=10.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = pd.Timestamp('2035-05-09')
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']

def load_ohlcv(assets=None, end=VISIBLE_END):
    out = {}
    for a in (assets or ASSETS):
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists():
            f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date']).sort_values('date')
        df = df[df['date'] <= end].set_index('date')
        out[a] = df
    return out

def load_macro(name, end=VISIBLE_END):
    df = pd.read_csv(INDEX_DIR / f'{name}.csv', parse_dates=['date']).sort_values('date')
    df = df[df['date'] <= end].set_index('date')
    return df['close'].astype(float)

def build_panel(end=VISIBLE_END):
    ohlcv = load_ohlcv(ASSETS, end)
    close, high, low, vol = {}, {}, {}, {}
    for a in ASSETS:
        df = ohlcv[a]
        close[a] = df['close'].astype(float)
        high[a] = df['high'].astype(float)
        low[a] = df['low'].astype(float)
        vol[a] = df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan, index=df.index)
    closes = pd.DataFrame(close)
    highs = pd.DataFrame(high).reindex(closes.index)
    lows = pd.DataFrame(low).reindex(closes.index)
    vols = pd.DataFrame(vol).reindex(closes.index)
    rets = closes.pct_change().dropna()
    return closes, highs, lows, vols, rets

def compute_ic(fv, fwd, min_dates=30, min_assets=8):
    common = sorted(set(fv.index) & set(fwd.index))
    ics, dates_ok = [], 0
    for d in common:
        f = fv.loc[d]; r = fwd.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= min_assets:
            dates_ok += 1
            x = f[m].rank().values; y = r[m].rank().values
            if np.std(x) > 0 and np.std(y) > 0:
                ics.append(np.corrcoef(x, y)[0, 1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return dict(IC=0.0, ICIR=0.0, n=len(ics), dates_ok=dates_ok, hit=0.0)
    mu = ics.mean(); sd = ics.std()
    return dict(IC=float(mu), ICIR=float(mu / sd * np.sqrt(len(ics)) if sd > 0 else 0.0),
                n=int(len(ics)), dates_ok=dates_ok, hit=float((ics > 0).mean()))

def coverage(fv):
    cov = float(fv.notna().sum().sum()) / (fv.shape[0] * fv.shape[1])
    dob = float((fv.notna().sum(axis=1) >= 8).mean())
    return cov, dob

closes, highs, lows, vols, rets = build_panel()

# forward returns at various horizons
fwd10 = rets.shift(-10).rolling(10).mean()
fwd5 = rets.shift(-5).rolling(5).mean()
fwd20 = rets.shift(-20).rolling(20).mean()

# macro series aligned to closes.index
vix = load_macro('VIX').reindex(closes.index)
dxy = load_macro('DXY').reindex(closes.index)
usdcny = load_macro('USDCNY').reindex(closes.index)
usdjpy = load_macro('USDJPY').reindex(closes.index)

cands = {}

# 1. Closing location within daily range (reversal/value at day level), smoothed 10d
hl_rng = (highs - lows).replace(0, np.nan)
cloc = ((closes - lows) / hl_rng).rolling(10).mean()
cands['cloc10'] = cloc

# 2. Cross-sectional dispersion regime: is ex-ante cross-sectional vol high -> favor momentum? (market-level, applied equal)
cs_rv = rets.std(axis=1).rolling(20).mean()
cands['xsec_rv_cs'] = cs_rv

# 3. downside-realized-vol ratio 20 (asymmetry of risk)
down = rets.clip(upper=0)
dd = (down**2).rolling(20).mean()**0.5
tot = rets.rolling(20).std()
cands['down_ratio20'] = dd / tot.replace(0, np.nan)

# 4. vol-of-vol 20x60 (regime change anticipation)
rv20 = rets.rolling(20).std()
cands['vol_of_vol'] = rv20.rolling(60).std() / rv20.rolling(60).mean()

# 5. VIX term spread proxy: VIX vs 20d realized of SPX ratio (already partly covered) - use VIX z-score
vix_z = (vix - vix
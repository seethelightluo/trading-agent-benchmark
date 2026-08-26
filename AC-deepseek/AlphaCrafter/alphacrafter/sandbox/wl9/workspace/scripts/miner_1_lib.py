"""miner_1 validation library (shared by candidate scripts).
Visible through 2033-08-17 (current cycle 2033-08-18). No lookahead: all
signals use data <= VISIBLE_END, forward returns measured after VISIBLE_END.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = pd.Timestamp('2033-11-09')
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
MACRO = ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']


def load_ohlcv(assets=None, end=VISIBLE_END):
    assets = assets or ASSETS
    out = {}
    for a in assets:
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


def build_panel(end=VISIBLE_END, assets=None):
    assets = assets or ASSETS
    ohlcv = load_ohlcv(assets, end)
    close, high, low, vol = {}, {}, {}, {}
    for a in assets:
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
        f = fv.loc[d]
        r = fwd.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= min_assets:
            dates_ok += 1
            x = f[m].rank().values
            y = r[m].rank().values
            if np.std(x) > 0 and np.std(y) > 0:
                ics.append(np.corrcoef(x, y)[0, 1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return dict(IC=0.0, ICIR=0.0, n=len(ics), dates_ok=dates_ok, hit=0.0)
    mu = ics.mean()
    sd = ics.std()
    return dict(IC=float(mu), ICIR=float(mu / sd * np.sqrt(len(ics)) if sd > 0 else 0.0),
                n=int(len(ics)), dates_ok=dates_ok, hit=float((ics > 0).mean()))


def coverage(fv):
    cov = float(fv.notna().sum().sum()) / (fv.shape[0] * fv.shape[1])
    dates_ok = float((fv.notna().sum(axis=1) >= 8).mean())
    return cov, dates_ok


def turnover(fv):
    r = fv.rank(axis=1)
    s = np.sign(r.sub(fv.shape[1] / 2)).fillna(0)
    return float((s.diff() != 0).mean().mean())


def decay_ic(fv, rets, horizons=(1, 2, 3, 5, 10, 20), min_dates=30):
    out = {}
    for h in horizons:
        fwd = rets.shift(-h).rolling(h).mean()
        r = compute_ic(fv, fwd, min_dates=min_dates)
        out[str(h)] = r['IC'] if r['n'] > 0 else 0.0
    return out


def report(name, fv, fwd5, fwd10, fwd20, verbose=True):
    a = compute_ic(fv, fwd10)
    b = compute_ic(fv, fwd5)
    c = compute_ic(fv, fwd20)
    ok = abs(a['IC']) >= 0.0070 and abs(a['ICIR']) >= 0.084
    if verbose:
        print(f"[{'OK' if ok else '--'}] {name:28s} IC={a['IC']:+.4f} ICIR={a['ICIR']:+.4f} "
              f"n={a['n']:4d} hit={a['hit']:.3f} | [5]{b['IC']:+.3f}[20]{c['IC']:+.3f}", flush=True)
    return a, ok
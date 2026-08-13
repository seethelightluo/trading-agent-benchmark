"""
miner3_harness.py - shared validation utilities for factor mining (miner_3).
Loads the 15-instrument cross-asset panel + macro observation data,
computes daily cross-sectional rank IC for a candidate factor, and reports
IC / ICIR / hit ratio / coverage / turnover / decay / per-year + regime stats.

Usage: import from a candidate script and call evaluate_factor().
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
         'COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO = {'DXY':'../persistent/index_data/DXY.csv',
         'USDCNY':'../persistent/index_data/USDCNY.csv',
         'USDJPY':'../persistent/index_data/USDJPY.csv',
         'EURUSD':'../persistent/index_data/EURUSD.csv',
         'VIX':'../persistent/index_data/VIX.csv'}
VISIBLE_THROUGH = '2032-04-30'   # data visible as of current date 2032-05-03
MIN_INSTR = 8                    # min instruments per date for a valid IC obs

_cache = {}
def load_panel(days=4500):
    """Return {symbol: DataFrame(date,open,high,low,close,volume)} for watchlist."""
    if 'panel' in _cache:
        return _cache['panel']
    panel = {}
    for s in WATCH:
        df = get_stock_daily_data(symbol=s, days=days)
        if df is None:
            continue
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        # keep only visible data
        df = df[df.index <= pd.Timestamp(VISIBLE_THROUGH)]
        panel[s] = df
    _cache['panel'] = panel
    return panel

def load_macro():
    """Return {name: Series(close)} for macro obs, truncated to visible window."""
    if 'macro' in _cache:
        return _cache['macro']
    out = {}
    for name, path in MACRO.items():
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[df.index <= pd.Timestamp(VISIBLE_THROUGH)]
        out[name] = df['close']
    _cache['macro'] = out
    return out

def factor_series(panel, fn, extra=None):
    """
    Apply fn(symbol, df) -> Series(factor value indexed by date) per instrument.
    Returns DataFrame F[date, symbol].
    """
    cols = {}
    for s, df in panel.items():
        try:
            ser = fn(s, df)
            ser = pd.Series(ser).dropna()
            ser.index = pd.to_datetime(ser.index)
            ser = ser[~ser.index.duplicated(keep='last')]
            cols[s] = ser
        except Exception as e:
            print(f'  [warn] {s} factor failed: {e}')
    F = pd.DataFrame(cols).sort_index()
    return F

def forward_returns(panel, h=10):
    """Forward h-day return per instrument on its own calendar. DataFrame R[date, symbol]."""
    cols = {}
    for s, df in panel.items():
        close = df['close']
        fr = close.shift(-h) / close - 1.0
        cols[s] = fr
    R = pd.DataFrame(cols).sort_index()
    return R

def daily_ic(F, R, min_instr=MIN_INSTR):
    """Daily cross-sectional Spearman IC between factor values and forward returns."""
    ics, dates, nobs = [], [], []
    common = F.index.intersection(R.index)
    for d in common:
        f = F.loc[d]
        r = R.loc[d]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_instr:
            continue
        if f[mask].nunique() < 3 or r[mask].nunique() < 3:
            continue
        ic = f[mask].rank().corr(r[mask].rank())
        if not np.isfinite(ic):
            continue
        ics.append(ic)
        dates.append(d)
        nobs.append(int(mask.sum()))
    return pd.Series(ics, index=pd.DatetimeIndex(dates)), np.array(nobs)

def summarize(ics, nobs, label=''):
    if len(ics) == 0:
        print(f'{label}: NO VALID IC OBS')
        return None
    ic_mean = ics.mean()
    ic_std = ics.std(ddof=1) if len(ics) > 1 else np.nan
    icir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
    hit = (np.sign(ics) == np.sign(ic_mean)).mean()
    n_dates = len(ics)
    avg_instr = nobs.mean()
    print(f'{label}: n_dates={n_dates} avg_instr={avg_instr:.1f} IC={ic_mean:.4f} '
          f'ICstd={ic_std:.4f} ICIR={icir:.4f} hit={hit:.3f}')
    return {'ic': float(ic_mean), 'icir': float(icir), 'hit': float(hit),
            'n_dates': int(n_dates), 'avg_instr': float(avg_instr),
            'ic_std': float(ic_std)}

def evaluate_factor(F, R, name='factor', horizons=(1, 3, 5, 10, 20), min_instr=MIN_INSTR):
    """Full evaluation: IC/ICIR per horizon + coverage + turnover + decay + regime splits."""
    print(f'===== EVALUATING: {name} =====')
    res = {}
    for h in horizons:
        Rh = forward_returns(load_panel(), h=h) if h != 10 else R
        ics, nobs = daily_ic(F, Rh, min_instr)
        s = summarize(ics, nobs, label=f'  h={h:2d}')
        if s:
            res[h] = s
            # per-year
            yrs = ics.groupby(ics.index.year)
            parts = []
            for y, g in yrs:
                if len(g) >= 20:
                    parts.append(f'{y}:IC={g.mean():+.4f}(n={len(g)})')
            print('      years:', ' '.join(parts))
    # coverage: fraction of instruments with factor value at each date (avg over dates)
    cov = F.notna().mean(axis=1)
    print(f'  coverage: mean={cov.mean():.3f} p10={cov.quantile(0.10):.3f} last50={cov.tail(50).mean():.3f}')
    # turnover: mean abs daily change of cross-sectional rank (normalized 0..1)
    ranks = F.rank(axis=1)
    rdiff = ranks.diff().abs().mean(axis=1)
    print(f'  rank-turnover: mean={rdiff.mean():.3f} (of max {2*(len(F.columns)-1)/len(F.columns):.3f})')
    # regime split by VIX tercile for h=10
    try:
        macro = load_macro()
        vix = macro['VIX'].reindex(F.index).ffill()
        q1, q2 = vix.quantile(0.33), vix.quantile(0.66)
        Rh = forward_returns(load_panel(), h=10)
        ics, _ = daily_ic(F, Rh, min_instr)
        for lab, m in [('LOW-VIX', vix <= q1), ('MID-VIX', (vix > q1) & (vix <= q2)), ('HIGH-VIX', vix > q2)]:
            sub = ics[m.reindex(ics.index).fillna(False)]
            if len(sub) >= 20:
                print(f'  regime[{lab}]: IC={sub.mean():+.4f} ICIR={sub.mean()/sub.std():+.3f} n={len(sub)}')
    except Exception as e:
        print(f'  regime split skipped: {e}')
    return res

def factor_file(factor_id):
    import json
    return f'factors/{factor_id}.json'

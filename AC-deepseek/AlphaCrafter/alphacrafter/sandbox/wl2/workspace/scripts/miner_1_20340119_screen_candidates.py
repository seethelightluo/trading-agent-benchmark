"""miner_1 2034-01-19 new candidate factor screen (horizon 10).

Motivation: re-validation shows long-horizon momentum (120/180d, 252d range)
has INVERTED vs 2026 admission, while volcluster_60 and spx_corr60 remain
positive. Screen interpretable candidates suited to the current rotation-heavy
regime: short-horizon reversal, vol-scaled 60d momentum, downside frequency,
cross-asset correlation tilts (XAU/WTI/ETH/BTC), drawdown recovery, and
vol-clustering variants.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

ASSETS = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
          'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
H = 10
MIN_ASSETS = 8
GATE_IC = 0.0070
GATE_ICIR = 0.0840

def load():
    out = {}
    for s in ASSETS:
        df = get_stock_daily_data(symbol=s, days=4000)
        if df is None or len(df) < 300:
            continue
        df = df.copy(); df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        out[s] = df
    idx = None
    for s, df in out.items():
        idx = df.index if idx is None else idx.union(df.index)
    idx = idx.sort_values()
    C = {s: df['close'].astype(float).reindex(idx) for s, df in out.items()}
    return pd.DataFrame(C)

def ic_series(fdf, fwd):
    ics, dates = [], []
    for dt in fdf.index:
        x = fdf.loc[dt]; y = fwd.loc[dt]
        m = x.notna() & y.notna()
        if m.sum() >= MIN_ASSETS:
            v = x[m].rank().corr(y[m].rank())
            if np.isfinite(v):
                ics.append(v); dates.append(dt)
    return pd.Series(ics, index=pd.DatetimeIndex(dates))

def summ(ic_s):
    if len(ic_s) < 5:
        return None
    ic = float(ic_s.mean()); icir = float(ic_s.mean()/ic_s.std())
    return {'ic': round(ic,4), 'icir': round(icir,3), 'hit': round(float((ic_s>0).mean()),3), 'n': len(ic_s),
            'pass': abs(ic) >= GATE_IC and abs(icir) >= GATE_ICIR}

def show(name, fdf, fwd, C):
    fdf = fdf.reindex(columns=C.columns)
    ic_s = ic_series(fdf, fwd)
    full = summ(ic_s)
    tr = ic_s[ic_s.index >= ic_s.index[-1] - pd.Timedelta(days=365)] if len(ic_s) else ic_s
    rec = summ(tr)
    cov = float(fdf.notna().mean().mean())
    if full:
        print(f"{name:28s} full ic={full['ic']:+.4f} icir={full['icir']:+.3f} hit={full['hit']:.3f} n={full['n']:4d} PASS={str(full['pass']):5s} | 365d ic={rec['ic'] if rec else float('nan'):+.4f} icir={rec['icir'] if rec else float('nan'):+.3f} | cov={cov:.3f}")
    return full, rec

def main():
    C = load(); R = C.pct_change()
    fwd = C.shift(-H) / C - 1.0
    print('grid', C.shape, C.index.min().date(), '->', C.index.max().date())

    cands = {}
    cands['reversal_5d'] = lambda C,R: -(C.shift(0)/C.shift(5)-1.0)
    cands['reversal_10d'] = lambda C,R: -(C/C.shift(10)-1.0)
    cands['reversal_20d'] = lambda C,R: -(C/C.shift(20)-1.0)
    cands['mom60_skip5'] = lambda C,R: C.shift(5)/C.shift(65)-1.0
    cands['mom60_vol60'] = lambda C,R: (C.shift(5)/C.shift(65)-1.0) / R.rolling(60).std()
    cands['mom90_skip5'] = lambda C,R: C.shift(5)/C.shift(95)-1.0
    cands['downside_freq_20'] = lambda C,R: (R < 0).rolling(20).mean()
    cands['downside_freq_60'] = lambda C,R: (R < 0).rolling(60).mean()
    cands['skew_20'] = lambda C,R: R.rolling(20).skew()
    cands['kurt_20'] = lambda C,R: R.rolling(20).kurt()
    cands['xau_corr60'] = lambda C,R: R.rolling(60, min_periods=15).corr(R['XAU'])
    cands['wti_corr60'] = lambda C,R: R.rolling(60, min_periods=15).corr(R['WTI'])
    cands['eth_corr60'] = lambda C,R: R.rolling(60, min_periods=15).corr(R['ETH'])
    cands['btc_corr60'] = lambda C,R: R.rolling(60, min_periods=15).corr(R['BTC'])
    cands['copper_corr60'] = lambda C,R: R.rolling(60, min_periods=15).corr(R['COPPER'])
    cands['ndx_corr60'] = lambda C,R: R.rolling(60, min_periods=15).corr(R['NDX'])
    cands['drawdown_60'] = lambda C,R: C/C.rolling(60).max()-1.0
    cands['drawdown_120'] = lambda C,R: C/C.rolling(120).max()-1.0
    cands['vol_ratio_5_60'] = lambda C,R: R.rolling(5).std()/R.rolling(60).std()
    cands['vol_ratio_20_120'] = lambda C,R: R.rolling(20).std()/R.rolling(120).std()
    cands['volcluster_20'] = lambda C,R: R.abs().rolling(20, min_periods=15).corr(R.abs().shift(1))
    cands['volcluster_40'] = lambda C,R: R.abs().rolling(40, min_periods=25).corr(R.abs().shift(1))
    cands['volcluster_60'] = lambda C,R: R.abs().rolling(60, min_periods=40).corr(R.abs().shift(1))
    cands['gain_loss_ratio_20'] = lambda C,R: (R.clip(lower=0).rolling(20).mean()) / (R.clip(upper=0).abs().rolling(20).mean() + 1e-9)
    cands['updown_vol_ratio_20'] = lambda C,R: (R.where(R>0).rolling(20).std()) / (R.where(R<0).rolling(20).std())
    cands['max_gain_20'] = lambda C,R: R.rolling(20).max()
    cands['max_loss_20'] = lambda C,R: R.rolling(20).min()
    cands['hl_range_20d_vol'] = lambda C,R: ((C.rolling(20).max()-C.rolling(20).min())/C) / R.rolling(20).std()
    cands['yield_spread_proxy'] = lambda C,R: C['US10Y'] / C - 1.0  # cheap proxy: asset vs US10Y level (na)
    cands['xau_x_us10y'] = lambda C,R: R['XAU'] * (C['US10Y'].pct_change())
    cands['crypto_x_equity'] = lambda C,R: R['BTC'] * R['SPX']

    for name, fn in cands.items():
        try:
            fdf = fn(C, R)
            if fdf is None:
                continue
            show(name, fdf, fwd, C)
        except Exception as e:
            print('ERR', name, str(e)[:80])

if __name__ == '__main__':
    main()

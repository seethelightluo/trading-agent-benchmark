"""miner_2 cycle 2033-04-28. Visible through 2033-04-27. No lookahead.
Revalidate effective 10-factor library and sweep novel candidates.
Gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084 (10d horizon),
15-asset cross-asset tradable universe. Report dates/instruments used.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2033-04-27'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets, end):
    C, H, L, V = {}, {}, {}, {}
    for a in assets:
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists():
            f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        C[a] = df['close'].astype(float)
        H[a] = df['high'].astype(float)
        L[a] = df['low'].astype(float)
        O[a] if a in O else None
        V[a] = df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan, index=df.index)
    return C, H, L, V

# open also needed for some candidates
def load_all(assets, end):
    C, H, L, Op, V = {}, {}, {}, {}, {}
    for a in assets:
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists():
            f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        C[a] = df['close'].astype(float)
        H[a] = df['high'].astype(float)
        L[a] = df['low'].astype(float)
        Op[a] = df['open'].astype(float) if 'open' in df else pd.Series(np.nan, index=df.index)
        V[a] = df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan, index=df.index)
    return C, H, L, Op, V

close, high, low, openp, vol = load_all(ASSETS, VISIBLE_END)
close = pd.DataFrame(close).dropna()
high = pd.DataFrame(high).reindex(close.index)
low = pd.DataFrame(low).reindex(close.index)
openp = pd.DataFrame(openp).reindex(close.index)
vol = pd.DataFrame(vol).reindex(close.index)
rets = close.pct_change().dropna()
idx = close.index
def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd5=fwd(5); fwd10=fwd(10); fwd20=fwd(20)
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d}", flush=True)

def mac(c):
    df = pd.read_csv(INDEX_DIR / f'{c}.csv', parse_dates=['date'])
    return df[df['date'] <= VISIBLE_END].set_index('date')['close'].astype(float).reindex(idx)
vix=mac('VIX'); usdcny=mac('USDCNY'); dxy=mac('DXY')
dVIX=vix.pct_change(); dCNY=usdcny.pct_change(); dxyr=dxy.pct_change()

def compute_ic(fv, fwd_, min_dates=30, start=None):
    f = fv.reindex(fwd_.index)
    ii = fwd_.index
    if start: ii = ii[ii >= pd.Timestamp(start)]
    ics=[]; ok=0
    for d in ii:
        x=f.loc[d]; y=fwd_.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8:
            ok+=1
            if np.std(x[m].rank().values)>0 and np.std(y[m].rank().values)>0:
                ics.append(np.corrcoef(x[m].rank(), y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates:
        return dict(ic=0.0, icir=0.0, n=len(ics), hit=0.0, cov=0.0, ok=ok)
    mu=ics.mean(); sd=ics.std()
    icir = mu/sd*np.sqrt(len(ics)) if sd>0 else 0
    return dict(ic=float(mu), icir=float(icir), n=len(ics),
                hit=float((ics>0).mean()), cov=float(f.notna().mean().mean()), ok=ok)

def turnover(fv):
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name, fv, start=None):
    a=compute_ic(fv, fwd10, start=start)
    b=compute_ic(fv, fwd5,
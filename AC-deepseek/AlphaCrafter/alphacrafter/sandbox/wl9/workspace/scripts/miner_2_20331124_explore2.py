import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data

watch = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DAYS = 2500
closes={}; highs={}; lows={}; vols={}
for s in watch:
    df = get_stock_daily_data(symbol=s, days=DAYS)
    df['date']=pd.to_datetime(df['date']); df=df.set_index('date').sort_index()
    closes[s]=df['close']; highs[s]=df['high']; lows[s]=df['low']; vols[s]=df['volume']
close_df=pd.DataFrame(closes); high_df=pd.DataFrame(highs); low_df=pd.DataFrame(lows)
ret=close_df.pct_change()

def compute_ic(factor, horizon=10):
    fwd = close_df.pct_change(horizon).shift(-horizon)
    rows=[]
    for dt,row in factor.iterrows():
        f=row.dropna(); fw=fwd.loc[dt]
        common=f.index.intersection(fw.dropna().index)
        if len(common)>=8:
            ic=np.corrcoef(f[common],fw[common])[0,1]
            if np.isfinite(ic): rows.append(ic)
    return np.array(rows)

def report(name, fac):
    for h in [5,10,20]:
        a=compute_ic(fac,h)
        if len(a)>0:
            print(f'{name} h{h}: IC={a.mean():.4f} ICIR={a.mean()/a.std():.4f} hit={(a>0).mean():.4f} n={len(a)}', end='  ')
    print()
    return a

# 10d reversal
rev10 = -ret.rolling(10).sum().shift(1)
report('rev10', rev10)
# 10d reversal ranking z of... combine with value
rev15 = -ret.rolling(15).sum().shift(1)
report('rev15', rev15)

# distance from recent low (pullback)
dist_low = close_df/close_df.rolling(20).min().shift(1) - 1   # positive when bouncing off low
report('dist_low_20', dist_low)

# inverse vol factor: low realized vol predicts...
rv = ret.rolling(20).std().shift(1)
report('invvol_20', -rv)

# momentum efficiency: |sum ret| / sum |ret| over 10d
absret = ret.abs()
rat = ret.rolling(10).sum().abs()/(absret.rolling(10).sum()+1e-12)
report('efficiency_10', rat.shift(1))

# range contraction / expansion (coiling): 20d vol rank / 60d vol
vol20=ret.rolling(20).std(); vol60=ret.rolling(60).std()
coil = (vol20/vol60).shift(1)
report('vol_ratio_20_60', -coil)
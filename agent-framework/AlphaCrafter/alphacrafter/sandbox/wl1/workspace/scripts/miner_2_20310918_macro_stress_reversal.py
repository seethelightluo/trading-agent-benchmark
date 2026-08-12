import os, sys
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s, days=5000)
            if x is not None and len(x): return x[['date','close']].copy()
        except Exception: pass
    return None
px={s:fetch(s) for s in U}
px={s:x for s,x in px.items() if x is not None}
wide=pd.concat([x.rename(columns={'close':s}).set_index('date') for s,x in px.items()],axis=1).sort_index().ffill()
v=get_index_daily_data('VIX',days=5000)
v=v[['date','close']].set_index('date').rename(columns={'close':'vix'}).sort_index().ffill()
d=wide.join(v,how='inner').loc['2020-01-01':'2031-09-03']
# macro stress: VIX relative to trailing 126d median, lagged; short reversal is negative 5d return,
# with stronger contrarian allocation during elevated but non-spiking VIX (clip avoids outlier domination)
r5=np.log(d[U]).diff(5); vol20=np.log(d[U]).diff().rolling(20).std()*np.sqrt(252)
stress=(d.vix/d.vix.rolling(126,min_periods=60).median()-1).clip(-.5,1.0)
raw=(-r5/(vol20+1e-8)).mul((1+stress.clip(lower=0)),axis=0)
f=raw.shift(1)
# forward 20d log returns, cross-sectional IC
fr=np.log(d[U].shift(-20)/d[U])
rows=[]
for dt in f.index:
    a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(a)>=8:
        rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15),'meanIC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
for h in [1,5,10,20]:
 frh=np.log(d[U].shift(-h)/d[U]); z=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],frh.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(a.iloc[:,0].corr(a.iloc[:,1]))
 z=pd.Series(z).dropna();print('h',h,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
 z=r.loc[a:b,'ic'];print(a,b,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
# turnover rank proxy
rank=f[U].rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().mean(axis=1).dropna().mean()
print('turnover_proxy',turn)
out=f[U].copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_2_20310918_macro_stress_reversal_signal.csv')

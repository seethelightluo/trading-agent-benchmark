import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            x=f(s,days=4000)
            if x is not None and len(x)>100: return x[['date','close']].copy()
        except Exception: pass
    return None
D={s:fetch(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
print('instruments',len(D),[s for s in U if s not in D])
# aligned close panel
P=pd.concat([x.set_index('date').close.rename(s) for s,x in D.items()],axis=1).sort_index().ffill()
# factor: low realized volatility, lagged 20d; use daily log return std * sqrt20, negative
r=np.log(P).diff(); vol=r.rolling(20).std().shift(1)
f=-vol
rows=[]
for h in [1,3,5,10]:
  ic=[]; ns=[]; dates=[]
  fr=P.shift(-h)/P-1
  for dt in f.index:
    a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(a)>=8:
      z=a.iloc[:,0].rank().corr(a.iloc[:,1].rank())
      if pd.notna(z): ic.append(z); ns.append(len(a)); dates.append(dt)
  q=pd.Series(ic,index=pd.to_datetime(dates)); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean(),'nobs',len(q),'avgN',np.mean(ns))
# turnover rank changes across adjacent dates
rr=f.rank(axis=1,pct=True); ch=(rr.diff().abs().mean(axis=1)>0.05).mean(); print('coverage',f.notna().mean().mean(),'turnover_proxy',ch)
# regimes for h10
h=10; fr=P.shift(-h)/P-1; vals=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  z=a.iloc[:,0].rank().corr(a.iloc[:,1].rank())
  if pd.notna(z): vals.append((dt,z))
q=pd.Series(dict(vals));
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2028')]:
 x=q.loc[lo:hi]; print('regime',lo,hi,x.mean(),len(x))
q.to_csv('scripts/miner_1_20280615_lowvol20_ic.csv')
# signal artifact
f.to_csv('scripts/miner_1_20280615_lowvol20_signal.csv')

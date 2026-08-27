import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            x=f(s, days=6000)
            if x is not None and len(x): return x
        except Exception: pass
    return None
px={}
for s in U:
    d=fetch(s)
    if d is not None: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
r=P.pct_change()
# Upside-consistency momentum: signed medium-term return, discounted by volatility,
# and strengthened when positive sessions dominate (all lagged one session).
ret=P/P.shift(20)-1
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
pos=r.gt(0).rolling(40,min_periods=30).mean()
raw=ret/(vol+1e-8)*(2*pos-1)
F=raw.shift(1)
print('rows',len(P),'instruments',len(P.columns),'range',P.index.min(),P.index.max())
for h in [5,10,20,40,60]:
    fw=P.shift(-h)/P-1
    vals=[]; ns=[]
    for dt in F.index:
        a=F.loc[dt]; b=fw.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    q=pd.Series(vals).dropna(); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean(),'dates',len(q),'avgN',np.mean(ns))
q=[]
for dt in F.index:
 a=F.loc[dt]; b=fw.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
q=pd.Series(q).dropna(); print('coverage',F.notna().mean().mean(),'turnover10',F.rank(pct=True).diff(10).abs().mean().mean(),'last',F.index[-1])
for a,b in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-12-31'),('2033-01-01','2035-06-20')]:
 z=[]
 for dt in F.loc[a:b].index:
  x=pd.concat([F.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
  if len(x)>=8:z.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
 z=pd.Series(z).dropna();print('regime',a,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(len(z)),'n',len(z))

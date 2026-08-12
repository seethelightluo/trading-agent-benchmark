import os, json
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=4200)
    if x is None or len(x)<300: x=get_index_daily_data(s,days=4200)
    if x is not None: D[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float)
p=pd.DataFrame(D).sort_index().ffill()
r=p.pct_change()
# risk-adjusted medium-term momentum: 20d return divided by trailing 20d volatility, lagged naturally
sig=(p.shift(1)/p.shift(21)-1)/(r.rolling(20).std().shift(1)*np.sqrt(20))
fwd=p.shift(-1)/p-1
rows=[]; sigrows=[]
for dt in sig.index:
    a=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([a,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
        sigrows.append([dt]+[a.get(s,np.nan) for s in U])
a=pd.DataFrame(rows,columns=['date','n','ic']).dropna()
q=a.ic
print('dates',len(a),'avgN',round(a.n.mean(),3),'universe',len(U))
print('IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(sig.notna().mean().mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),4))
for name,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-31','2026-01-01','2031-09-17')]:
 z=a[(a.date>=lo)&(a.date<=hi)].ic
 print(name,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
out='scripts/miner_2_20310918_voladj_momentum20_signal.csv'; pd.DataFrame(sigrows,columns=['date']+U).to_csv(out,index=False)
print('artifact',out)

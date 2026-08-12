import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=get_stock_daily_data(s,4200)
 if x is None or len(x)<300:x=get_index_daily_data(s,4200)
 if x is not None:D[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# blended 20/60 trend persistence, penalized by recent reversal; lag all inputs one day
f=((p.shift(1)/p.shift(61)-1)*0.7+(p.shift(1)/p.shift(21)-1)*0.3) - 0.25*(p.shift(1)/p.shift(6)-1)
y=p.shift(-1)/p-1
rows=[]; sr=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));sr.append([d]+[f.loc[d].get(s,np.nan) for s in U])
a=pd.DataFrame(rows,columns=['date','n','ic']);q=a.ic
print('dates',len(a),'avgN',round(a.n.mean(),3),'universe',len(U));print('IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4));print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),4))
for n,l,h in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-31','2026-01-01','2031-09-17')]:
 z=a[(a.date>=l)&(a.date<=h)].ic;print(n,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
out='scripts/miner_2_20310918_blended_trend_signal.csv';pd.DataFrame(sr,columns=['date']+U).to_csv(out,index=False);print('artifact',out)

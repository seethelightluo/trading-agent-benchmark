import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None:D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change();m=r.median(axis=1);rm=m.rolling(60,min_periods=40).var();beta=r.rolling(60,min_periods=40).cov(m).div(rm,axis=0);res=r.sub(beta.mul(m,axis=0));v=r.rolling(20,min_periods=15).std();f=-res.div(v+1e-8)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for nm,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[mask].ic;print(nm,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
f.to_csv('scripts/miner_3_20310123_beta_neutral_1d_signal.csv')

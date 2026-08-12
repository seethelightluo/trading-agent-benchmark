import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100:x=get_index_daily_data(s,days=3000)
 if x is not None:D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); disp=r.std(axis=1); gate=(disp>disp.rolling(60,min_periods=30).quantile(.65)).astype(float)
# Shock reversal: fade unusually large one-day moves, but only in high-dispersion sessions.
zscore=r/vol; f=(-zscore).mul(gate,axis=0);f=f.sub(f.median(axis=1),axis=0)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1:rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date');q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 q2=a.loc[m].ic;print(nm,len(q2),round(q2.mean(),6),round(q2.mean()/q2.std(ddof=1),6))
for h in [3,5,10]:
 y=p.pct_change(h).shift(-h);rr=[]
 for i in range(len(p)-h):
  zz=pd.concat([f.iloc[i].rename('f'),y.iloc[i].rename('y')],axis=1).dropna()
  if len(zz)>=8 and zz.f.nunique()>1:rr.append(zz.f.corr(zz.y))
 print('horizon',h,'IC',round(np.nanmean(rr),6),'n',len(rr))
f.to_csv('scripts/miner_1_20301017_shock_reversal_signal.csv')

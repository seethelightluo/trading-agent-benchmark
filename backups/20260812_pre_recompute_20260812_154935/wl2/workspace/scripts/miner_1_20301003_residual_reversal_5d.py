import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s,days=3000)
    if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
    if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); ret5=p.pct_change(5)
vol=r.rolling(20,min_periods=15).std(); disp=r.std(axis=1); disp[r.count(axis=1)<8]=np.nan
gate=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
resid=ret5.sub(ret5.median(axis=1),axis=0)
f=(-resid/vol).mul(gate,axis=0); f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',round(a.n.mean(),3),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
for h in [3,5,10]:
 ys=p.pct_change(h).shift(-h); rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),ys.iloc[i].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y))
 print('horizon',h,'IC',round(np.nanmean(rr),6),'n',len(rr))
f.to_csv('scripts/miner_1_20301003_residual_reversal_5d_signal.csv')

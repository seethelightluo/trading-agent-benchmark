import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=3000)
 if x is None or len(x)<100: x=get_index_daily_data(s,days=3000)
 if x is not None: D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Broad one-day residual reversal, strengthened (not zeroed) in observable fear/high-dispersion regimes.
base=-r.sub(r.median(axis=1),axis=0).div(vol)
disp=r.std(axis=1); dg=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
try:
 v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); v=v.set_index('date').close.astype(float).reindex(p.index).ffill(); vg=(v>v.rolling(60,min_periods=30).median()).astype(float)
except Exception: vg=pd.Series(0.,index=p.index)
f=base.replace([np.inf,-np.inf],np.nan).mul(1+0.75*dg.fillna(0)*vg.fillna(0),axis=0)
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),z.mean(),z.mean()/z.std(ddof=1))
for k in [3,5,10]:
 y=r.rolling(k).sum().shift(-k+1); o=[]
 for i in range(len(p)-k):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:o.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',k,'IC',np.nanmean(o),'n',len(o))
f.to_csv('scripts/miner_3_20301031_broad_vix_reversal_signal.csv')

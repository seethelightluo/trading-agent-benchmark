import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=3000)
 if d is not None:
  d=d.set_index('date'); D[s]=d[['open','high','low','close']].astype(float)
# volume-independent close-location pressure, averaged over five sessions; fade pressure
clv={s:((d.close-d.low)-(d.high-d.close))/(d.high-d.low).replace(0,np.nan) for s,d in D.items()}
clv=pd.DataFrame(clv).sort_index().ffill(); r=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index().ffill().pct_change()
pressure=clv.rolling(5,min_periods=5).mean(); disp=r.std(axis=1); gate=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
f=(-pressure.sub(pressure.median(axis=1),axis=0)).mul(gate,axis=0)
rows=[]
for i in range(len(r)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((r.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,m in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[m].ic; print(nm,len(z),z.mean(),z.mean()/z.std(ddof=1))
for k in [3,5,10]:
 y=r.rolling(k).sum().shift(-k+1); o=[]
 for i in range(len(r)-k):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:o.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',k,'IC',np.nanmean(o),'n',len(o))
f.to_csv('scripts/miner_1_20301031_clv_dispersion_reversal_signal.csv')

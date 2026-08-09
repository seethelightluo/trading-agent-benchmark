import pandas as pd, numpy as np
from pathlib import Path
root=Path('../persistent'); end=pd.Timestamp('2026-10-07'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s): return pd.read_csv(root/'stock_data'/(s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date').loc[:end]
d={s:load(s) for s in syms}; px=pd.concat({s:x.close for s,x in d.items()},axis=1); vol=pd.concat({s:x.volume for s,x in d.items()},axis=1)
# Volume-confirmed short momentum: trailing 5d return scaled by log volume surprise over trailing 20d.
vr=np.log1p(vol).sub(np.log1p(vol).rolling(20,min_periods=15).mean()).div(np.log1p(vol).rolling(20,min_periods=15).std())
f=px.pct_change(5)*vr; y=px.shift(-1).div(px)-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('factor=5d momentum x volume surprise; dates',len(q),'names',q.n.mean(),'coverage',q.n.sum()/(len(q)*15));print('IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=q.loc[a:b];print(a+'-'+b,len(z),z.ic.mean(),z.ic.mean()/z.ic.std())
for h in [5,10]:
 yy=px.shift(-h).div(px)-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,len(vals),np.mean(vals),np.mean(vals)/np.std(vals))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'end',q.index.max().date()); f.to_csv('scripts/miner_1_20261008_volume_confirmed_signal.csv',index_label='date')

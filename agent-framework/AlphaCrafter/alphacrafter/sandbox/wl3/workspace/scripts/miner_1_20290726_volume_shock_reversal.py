import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  q=d.set_index('date'); D[s]=q[['close','volume']].astype(float)
px=pd.concat({s:q.close for s,q in D.items()},axis=1).sort_index(); vol=pd.concat({s:q.volume for s,q in D.items()},axis=1).reindex(px.index)
r=np.log(px/px.shift(1)); lv=np.log1p(vol.clip(lower=0))
# Contrarian return over 3 days, strengthened by unusual volume, with all inputs lagged one completed day.
vs=(lv-lv.rolling(60,min_periods=20).mean())/lv.rolling(60,min_periods=20).std()
f=(-(r.rolling(3).sum())*(1+0.25*vs.clip(-2,2))).shift(1)
rows=[]
for i,dt in enumerate(px.index):
 vals=f.loc[dt]
 for h in (5,10):
  if i+h>=len(px): continue
  fr=np.log(px.iloc[i+h]/px.iloc[i]); z=pd.concat([vals,fr],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,h,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
x=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('dates',len(px),'instruments',len(D),'observations',len(x),'avgN',x.n.mean())
for h in (5,10):
 z=x[x.h==h]; print('horizon',h,'valid_dates',len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
 for name,a,b in [('2020_22','2020-01-01','2022-12-31'),('2023_25','2023-01-01','2025-12-31'),('2026_29','2026-01-01','2029-07-25'),('recent250',None,None)]:
  zz=z.tail(250) if name=='recent250' else z[(z.date>=b)&(z.date<=b if False else z.date<=b)]
  if len(zz): print(' ',name,len(zz),round(zz.ic.mean(),6),round(zz.ic.mean()/zz.ic.std(),6))
rr=f.rank(axis=1,pct=True); turn=rr.diff().abs().mean(axis=1).mean()
print('coverage',f.notna().sum().sum()/(len(f)*len(D)),'turnover_proxy',turn)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20290726_volume_shock_reversal_signal.csv',index=False); print('artifact rows',len(out))

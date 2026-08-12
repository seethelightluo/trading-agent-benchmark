import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<300: d=get_index_daily_data(s,4000)
 if d is not None and len(d): D[s]=d.set_index('date')
px=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index(); lr=np.log(px).diff(); mkt=lr.mean(axis=1)
R=lr.rolling(5).sum(); beta=lr.rolling(60).cov(mkt).div(mkt.rolling(60).var(),axis=0)
res=R-beta.mul(mkt.rolling(5).sum(),axis=0); vol=lr.rolling(30).std()*np.sqrt(5)+1e-9
base=-res/vol
# Continuous dispersion state: cross-sectional absolute one-day return dispersion,
# normalized by its trailing 60-session median, avoids a brittle binary gate.
disp=lr.sub(lr.mean(axis=1),axis=0).abs().mean(axis=1)
state=(disp/(disp.rolling(60,min_periods=30).median()+1e-9)).clip(0.25,3.0)
f=(base.mul(state,axis=0)).shift(1)
f=f.sub(f.median(axis=1),axis=0)
print('candidate dispersion_scaled_residual_reversal')
for h in [1,3,5,10]:
 rows=[]
 for i,dt in enumerate(px.index[:-h]):
  z=pd.concat([f.loc[dt],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna()
  if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
 print('H',h,'obs',len(q),'avgN %.2f'%q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean()))
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  y=q.loc[a:b].ic
  if len(y): print(a+'-'+b,len(y),'%.6f'%y.mean(),'%.6f'%(y.mean()/y.std()))
rr=f.rank(axis=1,pct=True)
print('dates',len(px),'instruments',len(D),'coverage %.4f'%(f.notna().sum().sum()/(len(f)*len(D))),'turnover %.4f'%(rr.diff().abs().mean(axis=1).mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300530_dispersion_scaled_residual_signal.csv',index=False)

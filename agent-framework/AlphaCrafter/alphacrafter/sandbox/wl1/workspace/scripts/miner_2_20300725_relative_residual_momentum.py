import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(symbol=s,days=4000)
 if d is not None: D[s]=d.sort_values('date').set_index('date')['close'].astype(float)
common=sorted(set.intersection(*[set(D[s].index) for s in U])); P=pd.DataFrame({s:D[s].reindex(common) for s in U},index=common).ffill()
R=P.pct_change(); fwd={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
r20=P.pct_change(20); med=r20.median(axis=1); residual=r20.sub(med,axis=0)
vol=R.rolling(40).std(); trend=P.pct_change(60)
sig=(residual/vol).where(trend>0).shift(1)
for h,y in fwd.items():
 a=[]; n=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); n.append(len(z))
 a=np.asarray(a,float); print(h,'IC %.6f ICIR %.6f n_dates %d avgN %.2f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),len(a),np.mean(n),np.mean(a>0)))
print('endpoint',pd.Timestamp(common[-1]).date(),'dates',len(common),'assets',len(U),'coverage',np.isfinite(sig).mean().mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for label,start in [('2026+','2026-01-01'),('2028+','2028-01-01'),('2029+','2029-01-01'),('2030','2030-01-01')]:
 a=[]
 for dt in sig.index[sig.index>=pd.Timestamp(start)]:
  z=pd.concat([sig.loc[dt],fwd[1].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.asarray(a); print(label,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1) if len(a)>1 else np.nan)
sig.reset_index().to_csv('scripts/miner_2_20300725_relative_residual_momentum_signal.csv',index=False)

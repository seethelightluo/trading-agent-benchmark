import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-11-24')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change()
# Candidate: medium-horizon trend quality, conditioned by lagged VIX regime.
# VIX is observation-only and used solely as a lagged macro state variable.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
ret60=P.pct_change(60)
down=r.where(r<0,0).rolling(60,min_periods=40).std()*np.sqrt(60)
cons=(r>0).rolling(60,min_periods=40).mean()
base=(ret60/(down+1e-12))*(cons-0.5)
vz=(vix-vix.rolling(120,min_periods=80).mean())/(vix.rolling(120,min_periods=80).std()+1e-12)
# In stressed volatility, favor quality/consistency and reduce noisy trend exposure; positive multiplier remains.
macro=(1-0.25*vz.clip(-1.5,1.5)).clip(0.5,1.5)
sig=base.mul(macro,axis=0).shift(1)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_3_20321125_macro_regime_quality_momentum_signal.csv',index=False)
for H in [10,20,30]:
 y=P.shift(-H)/P-1; vals=[]; ns=[]; dates=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); dates.append(d)
   if prev is not None:
    a=sig.loc[d].reindex(U); b=prev.reindex(U)
    turns.append(np.nanmean(np.abs((a-a.mean())/(a.abs().mean()+1e-12)-(b-b.mean())/(b.abs().mean()+1e-12))))
   prev=sig.loc[d]
 x=np.asarray(vals); print('H',H,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12)*np.sqrt(252),4),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),4),'coverage',round(sig.notna().mean().mean(),4))
 for label,lo in [('2028','2028-01-01'),('recent','2031-10-01')]:
  z=[v for d,v in zip(dates,vals) if d>=pd.Timestamp(lo)]; print(' ',label,'n',len(z),'IC',round(np.nanmean(z),6) if z else np.nan)

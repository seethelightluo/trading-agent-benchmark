import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-12-08')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change()
# Efficiency-adjusted trend: directional 40-session move divided by path length,
# scaled by inverse volatility and penalized by drawdown. All inputs are lagged.
net=P.pct_change(40)
path=r.rolling(40,min_periods=32).apply(lambda x: np.abs(x).sum(),raw=True)
eff=net/(path+1e-12)
vol=r.rolling(60,min_periods=45).std()*np.sqrt(40)
dd=P/P.rolling(80,min_periods=60).max()-1
# Favor persistent trends with shallow current drawdown; monotonic and interpretable.
sig=(eff/(vol+1e-12))*(1+dd.clip(-0.5,0))
sig=sig.shift(1)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_1_20321209_efficiency_trend_quality_signal.csv',index=False)
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
 for label,lo in [('2028','2028-01-01'),('recent365','2031-12-08'),('recent120','2032-06-12')]:
  z=[v for d,v in zip(dates,vals) if d>=pd.Timestamp(lo)]; print(' ',label,'n',len(z),'IC',round(np.nanmean(z),6) if z else np.nan,'ICIR',round(np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12)*np.sqrt(252),4) if len(z)>1 else np.nan)

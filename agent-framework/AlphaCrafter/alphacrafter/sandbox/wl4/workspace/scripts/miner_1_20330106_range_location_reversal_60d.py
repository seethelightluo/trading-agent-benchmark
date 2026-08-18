import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2033-01-05')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
# Range-location reversal: assets near the top of their trailing 60d range receive negative scores.
lo=P.rolling(60,min_periods=45).min(); hi=P.rolling(60,min_periods=45).max()
location=(P-lo)/(hi-lo+1e-12)
sig=(0.5-location).shift(1)
art='scripts/artifacts/miner_1_20330106_range_location_reversal_60d_signal.csv'
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv(art,index=False)
for H in [5,10,20,30]:
 y=P.shift(-H)/P-1; vals=[]; ns=[]; dates=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); dates.append(d)
   if prev is not None:
    a=sig.loc[d].reindex(U); b=prev.reindex(U); turns.append(np.nanmean(np.abs(a-b)))
   prev=sig.loc[d]
 x=np.asarray(vals); print('H',H,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),4),'coverage',round(sig.notna().mean().mean(),4))
 for label,lo2 in [('recent365','2032-01-01'),('recent120','2032-08-01')]:
  z=[v for d,v in zip(dates,vals) if d>=pd.Timestamp(lo2)]; print(label,'n',len(z),'IC',round(np.nanmean(z),6) if z else np.nan,'ICIR',round(np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),6) if z else np.nan)

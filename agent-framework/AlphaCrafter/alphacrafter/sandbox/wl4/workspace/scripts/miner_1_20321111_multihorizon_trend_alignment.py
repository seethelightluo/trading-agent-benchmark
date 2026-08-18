import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-11-10')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change(); v=r.rolling(60,min_periods=40).std()
components=[P.pct_change(h)/(v*np.sqrt(h)) for h in (10,20,60)]
sig=sum(components)/len(components)
sig=sig.rank(axis=1,pct=True)-0.5
sig=sig.shift(1)
yall={h:P.shift(-h)/P-1 for h in (10,20,30)}
for h,y in yall.items():
  ics=[]; ns=[]; prev=None; turns=[]
  for d in sig.index:
    q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
    if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
      ics.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
      if prev is not None: turns.append(np.nanmean(np.abs(sig.loc[d].values-prev.values)))
      prev=sig.loc[d]
  x=np.asarray(ics); sd=np.nanstd(x,ddof=1)
  print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'dailyICIR',round(np.nanmean(x)/(sd+1e-12),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),6))
  for label,lo in [('mid','2030-01-01'),('recent365','2031-05-01'),('recent120','2032-05-01')]:
    z=[]
    for d in sig.index[sig.index>=pd.Timestamp(lo)]:
      q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
      if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
    print(' ',label,len(z),round(np.nanmean(z),6) if z else np.nan,round(np.nanmean(z)/(np.nanstd(z,ddof=1)+1e-12),6) if len(z)>1 else np.nan)
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/artifacts/miner_1_20321111_multihorizon_trend_alignment_signal.csv',index=False)
print('artifact_rows',len(out),'coverage',round(sig.notna().mean().mean(),6))

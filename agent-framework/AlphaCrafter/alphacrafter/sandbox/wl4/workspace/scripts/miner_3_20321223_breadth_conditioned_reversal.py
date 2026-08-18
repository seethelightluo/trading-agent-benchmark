import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-12-22')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change(); r20=P.pct_change(20)
# Breadth of positive lagged 20d returns, used only as a regime multiplier.
breadth=(r20>0).mean(axis=1)
# Stress multiplier is larger in one-sided/breadth-extreme regimes; reversal is the hypothesis.
reg=(breadth-0.5).abs().rolling(120,min_periods=80).zscore() if False else None
mu=breadth.rolling(120,min_periods=80).mean(); sd=breadth.rolling(120,min_periods=80).std()
z=((breadth-mu)/(sd+1e-12)).clip(-2,2)
mult=(1+0.30*z.abs()).clip(1,1.6)
vol=r.rolling(60,min_periods=40).std()*np.sqrt(20)
resid=r20.sub(r20.mean(axis=1),axis=0)
sig=(-resid/(vol+1e-12)).mul(mult,axis=0).shift(1)
art='scripts/artifacts/miner_3_20321223_breadth_conditioned_reversal_signal.csv'
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv(art,index=False)
for H in [5,10,20,30]:
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
 for label,lo in [('2028','2028-01-01'),('recent','2031-10-01'),('last260','2031-12-01')]:
  z=[v for d,v in zip(dates,vals) if d>=pd.Timestamp(lo)]; print(label,'n',len(z),'IC',round(np.nanmean(z),6) if z else np.nan)

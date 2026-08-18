import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2032-12-22')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=P.pct_change(); r20=P.pct_change(20)
vol=r.rolling(60,min_periods=40).std()*np.sqrt(20)
residual=r20.sub(r20.mean(axis=1),axis=0)
base=-residual/(vol+1e-12)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
d20=dxy.pct_change(20); z=(d20-d20.rolling(120,min_periods=80).mean())/(d20.rolling(120,min_periods=80).std()+1e-12)
# Interpretable conditioning: amplify reversal when dollar trend is unusually stretched.
macro=(1+0.35*z.clip(-2,2).abs()).clip(1,1.7)
sig=base.mul(macro,axis=0).shift(1)
art='scripts/artifacts/miner_1_20321223_dxy_conditioned_residual_reversal_signal.csv'
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv(art,index=False)
for H in [5,10,20,30]:
 y=P.shift(-H)/P-1; vals=[]; ns=[]; dates=[]; turns=[]; prev=None
 for d in sig.index:
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q)); dates.append(d)
   if prev is not None:
    a=sig.loc[d].reindex(U); b=prev.reindex(U); turns.append(np.nanmean(np.abs((a-a.mean())/(a.abs().mean()+1e-12)-(b-b.mean())/(b.abs().mean()+1e-12))))
   prev=sig.loc[d]
 x=np.asarray(vals); print('H',H,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/(np.nanstd(x,ddof=1)+1e-12),6),'hit',round(np.mean(x>0),4),'turn',round(np.nanmean(turns),4),'coverage',round(sig.notna().mean().mean(),4))
 for label,lo in [('recent365','2031-07-01'),('recent120','2032-07-01')]:
  z2=[v for d,v in zip(dates,vals) if d>=pd.Timestamp(lo)]; print(label,'n',len(z2),'IC',round(np.nanmean(z2),6) if z2 else np.nan,'ICIR',round(np.nanmean(z2)/(np.nanstd(z2,ddof=1)+1e-12),6) if z2 else np.nan)

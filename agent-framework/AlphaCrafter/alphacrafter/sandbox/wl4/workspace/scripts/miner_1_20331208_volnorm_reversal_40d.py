import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index();r=px.pct_change()
sig=-(px/px.shift(40)-1)/(r.rolling(40,min_periods=25).std()*np.sqrt(40))
sig.reset_index().to_csv('scripts/artifacts/miner_1_20331208_volnorm_reversal_40d_signal.csv',index=False)
ics=[];ns=[];cov=[];turn=[];prev=None
for d in sig.index:
 x=sig.loc[d];y=(px.shift(-10)/px-1).loc[d];ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum());cov.append(ok.mean())
  if prev is not None:turn.append((x[ok].rank()-prev[ok].rank()).abs().sum()/(len(U)**2))
  prev=x
z=np.asarray(ics);print('factor=volnorm_reversal_40d dates',len(z),'avg_inst',np.mean(ns),'coverage',np.mean(cov),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turn))
for n in [260,520,780]:
 q=z[-n:];print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'dates',len(q))
for h in [5,10,20]:
 yy=px.shift(-h)/px-1;a=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8:a.append(spearmanr(sig.loc[d][ok],yy.loc[d][ok]).statistic)
 q=np.asarray(a);print('decay',h,q.mean(),q.mean()/q.std(ddof=1),len(q))

import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
base='../persistent/stock_data'; cutoff=pd.Timestamp('2034-06-23')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d
P=pd.DataFrame(px).sort_index().loc[:cutoff]; R=P.pct_change()
mom=P.pct_change(60).shift(1); breadth=(P.pct_change(20).shift(1)>0).mean(axis=1)
# breadth-conditioned contrarian 60d trend: positive breadth amplifies reversal
sig=-mom.mul((0.5+breadth),axis=0)
for h in [10,20,40,60]:
 fwd=P.shift(-h)/P-1; ics=[]; cov=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);cov.append(len(z)/15)
 a=np.asarray(ics); ranks=sig.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).dropna().mean()
 print('horizon dates IC ICIR coverage turnover',h,len(a),np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12),np.mean(cov),turn)
# 20d regime
fwd=P.shift(-20)/P-1
for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2034-06-23')]:
 a=[]
 for dt in sig.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(a);print('regime',lo,hi,len(a),np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12))
sig.to_csv('scripts/miner_2_20340623_breadth_conditioned_momentum_signal.csv')

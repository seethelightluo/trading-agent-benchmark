import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-08-07')
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:cut]
r=px.pct_change()
# Candidate: cross-sectional 10d reversal, activated only when cross-asset return dispersion is elevated.
ret10=px.pct_change(10)
disp=ret10.std(axis=1).rolling(20,min_periods=15).mean()
# percentile gate uses only trailing dispersion history, lagged one day
threshold=disp.rolling(252,min_periods=100).quantile(.60).shift(1)
gate=(disp.shift(1)>threshold).astype(float)
sig=(-ret10).shift(1).mul(gate,axis=0)
def run(h,start=None):
 y=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in sig.index:
  if start and d<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=np.array(vals)
 return len(x),float(np.mean(ns)),float(np.mean(x)),float(np.mean(x)/(np.std(x,ddof=1)+1e-12)*np.sqrt(len(x))),float(np.mean(x>0))
for h in [1,5,10,20]: print(h,'full',run(h),'recent260',run(h,'2029-08-08'))
rank=sig.rank(axis=1,pct=True); print('coverage',float(sig.notna().mean().mean()),'turnover',float(rank.diff().abs().mean(axis=1).mean()),'assets',px.shape[1],'dates',len(px),'cut',cut.date(),'gate_rate',float(gate.mean()))

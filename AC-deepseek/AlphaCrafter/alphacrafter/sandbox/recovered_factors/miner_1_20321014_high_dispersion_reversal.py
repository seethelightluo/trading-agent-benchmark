import pandas as pd,numpy as np,glob,os,json
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index().loc[:'2032-10-13']; R=P.pct_change()
# Candidate: high-dispersion shock reversal. Cross-sectional reversal is amplified only when
# contemporaneous cross-asset dispersion was elevated versus its trailing median.
disp=R.std(axis=1); scale=(disp/disp.rolling(120,min_periods=60).median()).clip(0,3)
F=R.rolling(3,min_periods=3).sum().mul(-1).mul(scale,axis=0).shift(1)
print('candidate high_dispersion_3d_reversal; dates',len(P),'assets',len(A),'coverage',round(F.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1; out=[];ns=[]
 for d in P.index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=np.array(out); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round(np.mean(s>0),4))
print('turnover10',round(F.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-10-13')]:
 vals=[]; y=P.shift(-1)/P-1
 for d in P.loc[lo:hi].index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals); print('REG',lo,hi,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
# save values for correlation audit
F.to_pickle('/tmp/candidate.pkl')

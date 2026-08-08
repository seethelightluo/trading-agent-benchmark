import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index(); R=P.pct_change()
# Persistent downside-variance share: low share means returns are less dominated by losses.
dn=R.where(R<0,0).pow(2).rolling(40,min_periods=25).mean(); tot=R.pow(2).rolling(40,min_periods=25).mean(); F=(1-dn/tot).shift(1)
print('data',P.index.min().date(),P.index.max().date(),'assets',len(A),'dates',len(P),'coverage',round(F.notna().mean().mean(),4))
def calc(y,F):
 out=[];ns=[]
 for d in P.index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: out.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=np.array(out);return len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),np.mean(s>0)
for h in [1,5,10,20]: print('H',h,*(round(x,6) if isinstance(x,float) else x for x in calc(P.shift(-h)/P-1,F)))
print('turnover10',round(F.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))
for lo,hi in [('2024','2027-12-31'),('2028','2030-12-31'),('2031','2032-07-15')]:
 sub=P.loc[lo:hi]; vals=[]; y=P.shift(-1)/P-1
 for d in sub.index:
  z=pd.concat([F.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals);print('REG',lo,hi,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))

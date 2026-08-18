import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-09-04')
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:cut]
r=px.pct_change(); mom=px.pct_change(20); vol=r.rolling(20,min_periods=15).std(); sig=(mom/(vol+1e-12)).shift(1)
def run(h,start=None):
 y=px.pct_change(h).shift(-h); vals=[];ns=[]
 for d in sig.index:
  if start and d<pd.Timestamp(start):continue
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 x=np.asarray(vals);return len(x),np.mean(ns),np.mean(x),np.mean(x)/(np.std(x,ddof=1)+1e-12),np.mean(x>0)
for h in [1,5,10,20]:print('h',h,'full',run(h),'recent261',run(h,'2029-09-05'))
rank=sig.rank(axis=1,pct=True);print('assets',px.shape[1],'dates',len(px),'coverage',sig.notna().mean().mean(),'avgN',sig.notna().sum(axis=1).mean(),'turnover',rank.diff().abs().mean(axis=1).mean())

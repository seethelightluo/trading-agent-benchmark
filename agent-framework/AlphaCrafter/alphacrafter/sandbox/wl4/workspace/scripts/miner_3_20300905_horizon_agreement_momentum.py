import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-09-04')
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:cut]
# Novel idea: trend agreement. Average standardized 5d and 20d returns, retain only assets whose horizons agree in sign.
r=px.pct_change(); r5=px.pct_change(5); r20=px.pct_change(20)
z5=r5.sub(r5.mean(axis=1),axis=0).div(r5.std(axis=1),axis=0)
z20=r20.sub(r20.mean(axis=1),axis=0).div(r20.std(axis=1),axis=0)
agree=np.sign(r5)==np.sign(r20)
sig=((z5+z20)/2).where(agree).shift(1)
def run(h,start=None):
 y=px.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in sig.index:
  if start and d<pd.Timestamp(start): continue
  q=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
 x=np.asarray(vals)
 return len(x),np.mean(ns),np.mean(x),np.mean(x)/(np.std(x,ddof=1)+1e-12),np.mean(x>0)
for h in [1,5,10,20]: print('h',h,'full',run(h),'recent261',run(h,'2029-09-05'))
print('assets',px.shape[1],'dates',len(px),'coverage',sig.notna().mean().mean(),'avgN',sig.notna().sum(axis=1).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'agreement',agree.mean().mean())

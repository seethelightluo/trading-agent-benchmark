import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E='2035-05-09'
def read(a,kind='stock'):
 p=('../persistent/stock_data/' if kind=='stock' else '../persistent/index_data/')+a+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.close,errors='coerce').loc[:E]
P=pd.concat([read(a).rename(a) for a in A],axis=1); R=P.pct_change()
# Low-volatility leadership: 20-session risk-adjusted trend, downweighted during VIX shocks.
v=read('VIX','index'); vshock=(v.pct_change(5).shift(1)).reindex(P.index)
cond=(1-vshock.clip(-.5,.5)).clip(.5,1.5)
F=R.rolling(20).sum().shift(1).div(R.rolling(20).std().shift(1)).mul(cond,axis=0)
F=F.replace([np.inf,-np.inf],np.nan)
def test(h):
 fr=P.pct_change(h).shift(-h); x=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 x=np.array(x); return len(x),np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1),np.mean(x>0),np.mean(ns)
for h in [1,5,10,20]:print('H',h,'dates IC ICIR hit meanN',test(h))
for lo,hi in [('2020','2025'),('2025','2030'),('2030','2033'),('2033','2035')]:
 fr=P.pct_change(10).shift(-10);x=[]
 for d in F.loc[lo:hi].index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(x);print('REG',lo,hi,len(x),np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1))
print('rows',len(P),'cells',int(F.count().sum()),'coverage',F.count().sum()/(len(F)*15),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
print('library_json_count',len(__import__('glob').glob('factors/*.json')),'max_abs_library_correlation','UNAVAILABLE: exact admitted signal panels are not persisted')

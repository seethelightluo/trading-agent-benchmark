import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  q=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[a]=q['close'];
prices=pd.DataFrame(D).sort_index(); rets=prices.pct_change()
# Volume-confirmed trend: prior 20d return scaled by log volume participation, all shifted one day
vol={}
for a in D:
 q=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 vol[a]=q['volume']
volume=pd.DataFrame(vol).reindex(prices.index)
participation=np.log1p(volume)/np.log1p(volume.rolling(60,min_periods=30).median())
f=(rets.rolling(20,min_periods=14).sum()*participation).shift(1)
print('candidate=volume_confirmed_trend_20obs')
print('dates',len(prices),'instruments',len(prices.columns),'coverage',round(f.notna().stack().mean(),6),'meanN',round(f.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 fr=prices.shift(-h)/prices-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(vals);print('h',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
fr=prices.shift(-10)/prices-1
for label,mask in [('2020-23',prices.index<'2024-01-01'),('2024-27',(prices.index>='2024-01-01')&(prices.index<'2028-01-01')),('2028+',prices.index>='2028-01-01'),('latest120',prices.index>=prices.index[-120])]:
 a=[]
 for dt in prices.index[mask]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.asarray(a);print('regime',label,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6))
for name,c in [('ret20',rets.rolling(20,min_periods=14).sum()),('participation',participation)]:
 z=pd.concat([f.stack().rename('f'),c.shift(1).stack().rename('c')],axis=1).dropna(); print('corr',name,round(spearmanr(z.f,z.c).statistic,6),'cells',len(z))

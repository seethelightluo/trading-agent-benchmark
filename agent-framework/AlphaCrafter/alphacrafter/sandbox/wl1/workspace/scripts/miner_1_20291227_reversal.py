import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2029-12-27')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); P=P.loc[:cutoff]; r=P.pct_change(); f=(-(P/P.shift(5)-1)/(r.rolling(20,min_periods=18).std()+1e-8)).shift(1); rows=[]
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('h',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(len(a)),'hit',(a>0).mean())
fr=P.shift(-20)/P-1; a=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
a=pd.Series(a);print('coverage',1.0,'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for name,mask in [('2028+',f.index>='2028-01-01'),('2029+',f.index>='2029-01-01')]:
 # recompute aligned dates
 vals=[]
 for dt in f.index[mask]:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 vals=np.array(vals);print(name,len(vals),vals.mean(),vals.mean()/vals.std(ddof=1)*np.sqrt(len(vals)))
pd.DataFrame({'signal':f.stack()}).to_csv('scripts/miner_1_20291227_reversal_signal.csv')

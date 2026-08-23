import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-03-17'); xs={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); xs[s]=d.loc[:end,'close'].astype(float)
p=pd.DataFrame(xs).sort_index(); r=np.log(p).diff(); f=pd.DataFrame(index=p.index)
for s in U:
 rr=r[s]; f[s]=-(rr.rolling(3).sum()-0.15*rr.rolling(20).sum())/rr.rolling(30).std().shift(1)
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1)*np.sqrt(252),'hit',(a>0).mean())
fr=p.shift(-10)/p-1; rows=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(q)>=8: rows.append((dt,len(q),spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); f.to_csv('scripts/miner_1_20330317_fast_reversal_signal.csv'); a.to_csv('scripts/miner_1_20330317_fast_reversal_ic.csv')

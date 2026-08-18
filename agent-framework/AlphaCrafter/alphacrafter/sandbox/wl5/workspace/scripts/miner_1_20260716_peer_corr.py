import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().loc[:'2026-07-15']
for lb in [3,5]:
 r=p.pct_change(lb); a=[]
 for dt in r.index:
  x=r.loc[dt]; f=pd.Series({s:x.drop(labels=s).median() for s in U})
  q=pd.concat([f.rename('peer'),(-x).rename('rev')],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.peer,q.rev).statistic)
 print(lb,'dates',len(a),'mean daily corr',np.mean(a),'mean abs',np.mean(np.abs(a)),'min',np.min(a))

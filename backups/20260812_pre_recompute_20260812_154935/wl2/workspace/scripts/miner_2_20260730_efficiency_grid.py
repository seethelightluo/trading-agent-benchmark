import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in D.values()])); C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U});R=C.pct_change()
for w in [10,12,15,20,25,30,40,60]:
 F=((C/C.shift(w)-1)/(R.abs().rolling(w,min_periods=max(8,w-2)).sum()+1e-12)).shift(1);Y=C.shift(-1)/C-1;a=[];ns=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 a=np.array(a);print('w',w,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'coverage',round(F.notna().sum().sum()/F.size,4),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))

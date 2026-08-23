import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2032-06-09'
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}).sort_index().loc[:cut]; r=P.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill(); ret=P/P.shift(20)-1; down=np.sqrt((r.clip(upper=0)**2).rolling(40,min_periods=20).mean())*np.sqrt(20); stress=(v/v.rolling(60,min_periods=40).median()).clip(.5,2); F=ret.div(down).mul(stress,axis=0)
print('cutoff',P.index.max().date(),'dates',len(P),'assets',P.shape[1],'coverage',round(F.notna().stack().mean(),4))
for h in [5,10,20]:
 a=[]; n=[]; ds=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.x,z.y).statistic);n.append(len(z));ds.append(P.index[i])
 a=np.array(a);print({'horizon':h,'valid_dates':len(a),'avg_instruments':round(np.mean(n),3),'IC':round(np.mean(a),6),'ICIR':round(np.mean(a)/np.std(a,ddof=1),6),'hit_ratio':round(np.mean(a>0),4)})
 if h==20: print('regimes',pd.DataFrame({'ic':a},index=ds).groupby(lambda x:x.year).ic.agg(['mean','count']).round(6).to_dict('index'))

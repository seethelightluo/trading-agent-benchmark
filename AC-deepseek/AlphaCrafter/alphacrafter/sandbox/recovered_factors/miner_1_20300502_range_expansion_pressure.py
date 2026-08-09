import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 rng=(d.high-d.low).replace(0,np.nan); clv=(2*d.close-d.high-d.low)/rng
 # pressure is recent close-location persistence, gated by range expansion vs prior baseline
 pressure=clv.rolling(5,min_periods=4).mean()
 expansion=(rng.rolling(5,min_periods=4).mean()/rng.rolling(40,min_periods=25).mean()).clip(0.25,4)
 D[a]=pressure* np.log(expansion).shift(1)
prices=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}).sort_index()
f=pd.DataFrame(D).reindex(prices.index).shift(1)
print('assets',len(assets),'period',f.index.min().date(),f.index.max().date(),'coverage',f.notna().stack().mean(),'meanN',f.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 fw=prices.shift(-h)/prices-1; vals=[]; ds=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds);print('H',h,'dates',len(s),'meanN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
 for label,sub in [('2020-24',s.loc[:'2024-12-31']),('2025-27',s.loc['2025':'2027']),('2028-30',s.loc['2028':'2030']),('latest120',s.iloc[-120:])]:
  print(' ',label,len(sub),'IC',sub.mean(),'ICIR',sub.mean()/sub.std(ddof=1) if len(sub)>1 else np.nan)
print('turnover10',f.rank(axis=1,pct=True).diff(10).abs().mean().mean())

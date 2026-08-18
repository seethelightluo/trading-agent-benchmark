import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
# prior-session intraday pressure: fade unusually positive close/open moves, normalized by recent intraday volatility
F=[]
for s,d in D.items():
 x=np.log(d.close/d.open); vol=x.rolling(20,min_periods=15).std(); f=(-x/vol).rename(s); F.append(f)
F=pd.concat(F,axis=1); out={}
for h in [1,5,10]:
  ic=[]; ns=[]; years={}
  for dt in F.index:
    xs=[];ys=[]
    for s in U:
      if dt not in D[s].index or pd.isna(F.loc[dt,s]): continue
      i=D[s].index.get_loc(dt); j=i+h
      if j<len(D[s]) and pd.notna(D[s].close.iloc[j]): xs.append(F.loc[dt,s]); ys.append(D[s].close.iloc[j]/D[s].close.iloc[i]-1)
    if len(xs)>=8 and len(set(xs))>1:
      q=spearmanr(xs,ys).statistic; ic.append(q);ns.append(len(xs));years.setdefault(str(dt.year),[]).append(q)
  a=np.array(ic); out[h]=(len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),{k:round(np.mean(v),4) for k,v in years.items()})
  print(h,out[h])
print('coverage',F.notna().sum(axis=1).mean()/15,'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2)
print('dates',F.index.min(),F.index.max())

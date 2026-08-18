import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
# Volatility shock: recent 5d realized volatility relative to 60d baseline; high shocks tend to mean-revert.
R=pd.concat({s:D[s].close.pct_change() for s in U},axis=1).sort_index()
short=R.rolling(5,min_periods=4).std(); base=R.rolling(60,min_periods=40).std()
F=-(short/base) # higher score = calmer / less shock, defensive interpretation
for h in [1,5,10]:
  ics=[]; ns=[]; yrs={}
  for dt in F.index:
    xs=[];ys=[]
    for s in U:
      if dt not in D[s].index or pd.isna(F.loc[dt,s]): continue
      ix=D[s].index.get_loc(dt); j=ix+h
      if j<len(D[s]) and pd.notna(D[s].close.iloc[j]): xs.append(F.loc[dt,s]);ys.append(D[s].close.iloc[j]/D[s].close.iloc[ix]-1)
    if len(xs)>=8 and len(set(xs))>1:
      q=spearmanr(xs,ys).statistic
      if pd.notna(q): ics.append(q);ns.append(len(xs));yrs.setdefault(dt.year,[]).append(q)
  a=np.asarray(ics); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'years',{k:round(np.mean(v),4) for k,v in yrs.items()})
print('coverage',round(F.notna().sum(axis=1).mean()/15,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
print('corr rev5',round(F.stack().corr((-R.rolling(5).sum()).stack()),4),'corr mom20',round(F.stack().corr(R.rolling(20).sum().stack()),4))

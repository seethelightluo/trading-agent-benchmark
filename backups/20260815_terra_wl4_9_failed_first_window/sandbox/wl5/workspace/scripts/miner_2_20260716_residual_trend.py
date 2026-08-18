import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
P=pd.DataFrame({s:D[s].close for s in U}); R=P.pct_change(); basket=R.mean(axis=1)
F=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 beta=R[s].rolling(60,min_periods=40).cov(basket)/basket.rolling(60,min_periods=40).var()
 F[s]=R[s].rolling(20,min_periods=15).sum()-beta*basket.rolling(20,min_periods=15).sum()
for h in [1,5,10]:
  ics=[];ns=[];yrs={}
  for i,dt in enumerate(F.index[:-h]):
    xs=[];ys=[]
    for s in U:
      if pd.isna(F.loc[dt,s]) or dt not in D[s].index: continue
      ix=D[s].index.get_loc(dt); j=ix+h
      if j<len(D[s]) and pd.notna(D[s].close.iloc[j]):xs.append(F.loc[dt,s]);ys.append(D[s].close.iloc[j]/D[s].close.iloc[ix]-1)
    if len(xs)>=8 and len(set(xs))>1:
      q=spearmanr(xs,ys).statistic;ics.append(q);ns.append(len(xs));yrs.setdefault(str(dt.year),[]).append(q)
  a=np.array(ics);print('h',h,'N',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'years',{k:round(np.mean(x),4) for k,x in yrs.items()})
print('coverage',round(F.notna().sum().sum()/(len(F)*15),4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
for n,p in [('rev5',-R.rolling(5).sum()),('mom20',R.rolling(20).sum())]: print('corr',n,round(F.stack().corr(p.stack()),4))

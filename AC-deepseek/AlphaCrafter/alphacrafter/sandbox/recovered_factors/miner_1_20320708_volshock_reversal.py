import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[a]=d['close']
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# volatility-shock reversal: reverse recent 3-day move, normalized by trailing 20d vol; all trailing data
vol=ret.rolling(20,min_periods=15).std()
factor=-(ret.rolling(3,min_periods=3).sum()/vol).shift(0)
# use observations through t, forward close return t to t+h (next h sessions)
for h in [1,5,10,20]:
 f=factor
 fr=close.shift(-h)/close-1
 ics=[]; ns=[]; turnovers=[]
 for dt in close.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   ics.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
 # rank turnover every 10 sessions
 dates=f.index
 for i in range(10,len(dates),10):
  d0,d1=dates[i-10],dates[i]; common=f.loc[[d0,d1]].dropna(axis=1).columns
  if len(common)>=8:
   turnovers.append(np.mean(f.loc[d0,common].rank().values!=f.loc[d1,common].rank().values))
 arr=np.array(ics); print('H',h,'dates',len(arr),'meanN',round(np.mean(ns),2),'IC',round(np.nanmean(arr),6),'ICIR',round(np.nanmean(arr)/np.nanstd(arr,ddof=1),6),'hit',round(np.mean(arr>0),4),'turn',round(np.mean(turnovers),4))
print('cells',int(factor.notna().sum().sum()),'coverage',factor.notna().mean().mean(),'dates',close.index.min(),close.index.max())
# regimes for best likely h
for lo,hi in [('2024-01-01','2027-12-31'),('2028-01-01','2030-12-31'),('2031-01-01','2032-07-07')]:
 vals=[]
 fr=close.shift(-10)/close-1
 for dt in close.loc[lo:hi].index:
  ok=factor.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(factor.loc[dt,ok],fr.loc[dt,ok]).statistic)
 print('regime',lo, len(vals), round(np.mean(vals),6),round(np.mean(vals)/np.std(vals,ddof=1),6))

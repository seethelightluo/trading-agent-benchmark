import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
u=r.mean(axis=1); ur=(1+u).rolling(20).apply(np.prod,raw=True)-1
ret20=p.pct_change(20); fac=ret20.sub(ur,axis=0)
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; ics=[]; turns=[]; cells=0
 for i in range(len(p)-h):
  x=fac.iloc[i]; y=fwd.iloc[i]; m=x.notna()&y.notna()
  if m.sum()>=8:
   ics.append(spearmanr(x[m],y[m]).statistic); cells+=m.sum()
   if i>0:
    prev=fac.iloc[i-1]; mm=m&prev.notna()
    if mm.sum()>=8: turns.append(np.mean(pd.Series(x[mm]).rank().values != pd.Series(prev[mm]).rank().values))
 print('H',h,'dates',len(ics),'IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(np.array(ics)>0),'turn',np.nanmean(turns),'cells',cells,'coverage',cells/(len(ics)*15))
fwd=p.shift(-10)/p-1
for yr in range(2020,2027):
 vals=[]
 for i,dt in enumerate(p.index):
  if dt.year!=yr: continue
  x=fac.iloc[i]; y=fwd.iloc[i];m=x.notna()&y.notna()
  if m.sum()>=8: vals.append(spearmanr(x[m],y[m]).statistic)
 print('YEAR',yr,len(vals),np.nanmean(vals) if vals else np.nan)

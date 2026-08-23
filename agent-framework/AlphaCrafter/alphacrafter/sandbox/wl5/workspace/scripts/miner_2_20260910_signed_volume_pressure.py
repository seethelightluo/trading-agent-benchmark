import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15')
def load(s):
 p='../persistent/stock_data/'+s+'.csv'; d=pd.read_csv(p); d.date=pd.to_datetime(d.date); d=d[d.date<=END].drop_duplicates('date').sort_values('date').set_index('date'); return d[['close','volume']].astype(float)
D={s:load(s) for s in U}; dates=sorted(set.intersection(*[set(x.index) for x in D.values()]));
p=pd.DataFrame({s:D[s].reindex(dates).close for s in U}); v=pd.DataFrame({s:D[s].reindex(dates).volume for s in U}); r=p.pct_change()
# Signed volume pressure: recent signed returns weighted by abnormal volume, normalized cross-sectionally.
volratio=(v/v.rolling(20,min_periods=15).mean()).clip(0,10)
f=(r*volratio).rolling(10,min_periods=7).sum()
f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10]:
 fw=p.pct_change(h).shift(-h); ics=[]; ns=[]
 for dt in f.index:
  z=pd.DataFrame({'f':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: ics.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
 a=np.asarray(ics); print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
print('coverage',round(f.notna().sum().sum()/f.size,5),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),5),'period',f.index.min(),f.index.max())
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15')]:
 a=[]
 for dt in f.loc[lo:hi].index:
  z=pd.DataFrame({'f':f.loc[dt],'y':p.pct_change().shift(-1).loc[dt]}).dropna()
  if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic)
 a=np.asarray(a);print('regime',lo[:4]+'-'+hi[:4],'dates',len(a),'ICIR',round(a.mean()/a.std(ddof=1),5) if len(a)>1 else None)

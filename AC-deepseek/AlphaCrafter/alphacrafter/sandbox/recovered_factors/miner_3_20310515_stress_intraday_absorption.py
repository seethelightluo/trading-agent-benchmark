import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
END=pd.Timestamp('2031-05-14')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
ix={'000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'}
# all benchmark files are in persistent stock_data in research environment
P={}; O={}; H={}; L={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+a+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
 P[a]=pd.to_numeric(d['close'],errors='coerce'); O[a]=pd.to_numeric(d['open'],errors='coerce'); H[a]=pd.to_numeric(d['high'],errors='coerce'); L[a]=pd.to_numeric(d['low'],errors='coerce')
P=pd.DataFrame(P).loc[:END]; O=pd.DataFrame(O).reindex(P.index); H=pd.DataFrame(H).reindex(P.index); L=pd.DataFrame(L).reindex(P.index)
# Stress-conditioned intraday absorption: pressure, residualized by cross-sectional median, only when VIX is elevated and rising.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
stress=(v>v.rolling(120,min_periods=60).quantile(.70)) & (v.diff(5)>0)
rng=(H-L).replace(0,np.nan)
raw=((P-O)/rng).rolling(10,min_periods=8).mean()
F=raw.sub(raw.median(axis=1),axis=0).where(stress, np.nan)
print('candidate stress dates',int(stress.sum()),'total',len(P),'mean valid',F.count(axis=1).where(stress).mean())
for h in [1,5,10,20]:
 R=P.pct_change(h).shift(-h)
 ics=[]; ns=[]
 for dt in F.index:
  x=F.loc[dt]; y=R.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 q=pd.Series(ics).dropna(); print('H',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
# annual/regime and turnover
R=P.pct_change().shift(-1); q=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
 if len(z)>=8:q.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(q,columns=['date','ic']).set_index('date'); print('regimes')
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2031')]:
 z=q.loc[lo:hi,'ic'];print(lo,len(z),round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('coverage all',round(F.notna().mean().mean(),4),'active turnover',F.notna().mean(axis=1).where(stress).mean())
# rank signal correlations to simple admitted proxies, explicit evidence (candidate against known constructs)
proxies={
 'ravmom20':P.pct_change(20).div(P.pct_change().rolling(20).std()),
 'volnormrev5':-P.pct_change(5).div(P.pct_change().rolling(20).std()),
 'trendcons20':P.pct_change().rolling(20).mean().div(P.pct_change().rolling(20).std()),
 'rangepos40':(P-P.rolling(40).min()).div(P.rolling(40).max()-P.rolling(40).min())}
mx=0; arg=''
for n,x in proxies.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); c=abs(spearmanr(z.f,z.x).statistic) if len(z)>20 else np.nan
 print('corr',n,round(c,6),len(z));
 if c>mx:mx,arg=c,n
print('max_abs_library_correlation',round(mx,6),arg)

import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); P=P[P.index<=cut]
R=P.pct_change(); m=R.median(axis=1)
# Relative momentum: asset 20d return less contemporaneous cross-asset median 20d return, smoothed only through date t
raw=P.pct_change(20).sub(P.pct_change(20).median(axis=1),axis=0)
F=raw.rolling(3,min_periods=3).mean()
print('rows',len(P),'dates',P.index.min(),P.index.max())
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.f.corr(z.y,method='spearman'))); ns.append(len(z))
 a=pd.DataFrame(vals,columns=['date','ic']).set_index('date').ic
 print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==1:
  for yr,g in a.groupby(a.index.year): print('year',yr,'dates',len(g),'IC',round(g.mean(),6),'ICIR',round(g.mean()/g.std(ddof=1),4))
print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for start in ['2024-01-01','2025-01-01','2026-01-01']:
 Y=P.pct_change().shift(-1); z=[]
 for dt in P.loc[start:].index:
  x=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
  if len(x)>=8:z.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'))
 z=pd.Series(z); print('recent',start,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
# correlations with common library proxies
for name,x in {'mom20':P.pct_change(20),'rev5':-P.pct_change(5),'clv':(P-P.rolling(20).min())/(P.rolling(20).max()-P.rolling(20).min())}.items():
 cs=[]
 for dt in P.index:
  q=pd.concat([F.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(q)>=8:cs.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('corr',name,round(np.nanmean(cs),4))

import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-12-16')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']; P[s]=d[d.index<=cutoff]
p=pd.DataFrame(P); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); fac=-vol
for h in [1,5,10]:
 y=p.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in p.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 q=np.array(vals); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6),'hit',round(np.mean(q>0),4))
 if h==1:
  ser=pd.Series(q,index=dates)
  print('years',[(int(y),round(g.mean(),6),len(g)) for y,g in ser.groupby(lambda x:x.year)])
print('coverage',round(np.mean(ns)/15,4),'period',p.index.min().date(),p.index.max().date())
# turnover on percentile ranks
prev=None;ts=[]
for dt in p.index:
 z=fac.loc[dt].dropna()
 if len(z)>=8:
  rr=z.rank(pct=True)
  if prev is not None:
   a=prev.index.intersection(rr.index)
   if len(a)>=8:ts.append(np.mean(abs(rr[a]-prev[a])))
  prev=rr
print('turnover',round(np.mean(ts),6))

import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-09-10')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').set_index('date').close for s in U}; p=pd.DataFrame(P).sort_index(); r=p.pct_change(); raw=p.pct_change(20); peer=raw.sub(raw.median(axis=1),axis=0); fac=peer/(r.rolling(40,min_periods=30).std()*np.sqrt(20))
print('DATA',len(p),len(U),p.index.min().date(),p.index.max().date())
for h in [1,5,10,20]:
 q=[]; ns=[]; dates=[]
 for i in range(45,len(p)-h):
  x=fac.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1; ok=x.notna()&y.notna()&np.isfinite(x)&np.isfinite(y)
  if ok.sum()>=8:q.append(spearmanr(x[ok],y[ok]).statistic);ns.append(ok.sum());dates.append(p.index[i])
 q=np.array(q); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5),'hit',round((q>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  a=q[[lo<=d.year<=hi for d in dates]]; print(' regime',lo,hi,len(a),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5))
print('turnover',round(np.nanmean(np.abs(fac.rank(pct=True).diff()).stack()),5))
for other in [raw,-p.pct_change(5)]:
 a,b=fac.align(other);ok=np.isfinite(a)&np.isfinite(b);print('corr',a[ok].corr(b[ok]))

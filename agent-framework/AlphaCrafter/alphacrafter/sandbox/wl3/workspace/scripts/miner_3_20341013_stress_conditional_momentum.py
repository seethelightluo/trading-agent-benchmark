import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 f=f'{base}/{s}.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d.date=pd.to_datetime(d.date); px[s]=d.set_index('date').close
P=pd.DataFrame(px).sort_index(); P=P.loc[P.index<=pd.Timestamp('2034-10-13')]; v=pd.read_csv('../persistent/index_data/VIX.csv'); v.date=pd.to_datetime(v.date); V=v.set_index('date').close.reindex(P.index).ffill()
m=P.pct_change(20); med=m.median(axis=1); rel=m.sub(med,axis='index')
stress=(V>V.rolling(120,min_periods=60).quantile(.70)).astype(float)
F=(rel.mul(1-2*stress,axis='index')).shift(1)
print('assets',len(px),'dates',len(P),'avg_valid',round(F.notna().sum(axis=1).mean(),3))
allrows={}
for h in [1,3,5,10,20]:
 vals=[]; dates=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(P.index[i])
 a=np.array(vals); ic=np.nanmean(a); ir=ic/np.nanstd(a,ddof=1)*np.sqrt(len(a))
 allrows[h]=(a,dates); print('h',h,'n',len(a),'IC',round(ic,6),'ICIR',round(ir,4),'hit',round(np.mean(a>0),4))
a,dates=allrows[10]
for n in [120,252,756,1260]:
 q=a[-n:]; print('recent',n,'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(len(q)),4))
r=F.rank(axis=1,pct=True); print('turnover',round(np.nanmean(np.abs(r-r.shift(1)).mean(axis=1)),4),'start',dates[0],'end',dates[-1])
for name,mask in [('stress',stress.loc[pd.to_datetime(dates)].values.astype(bool)),('normal',~stress.loc[pd.to_datetime(dates)].values.astype(bool))]:
 q=a[mask]; print(name,'n',len(q),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1)*np.sqrt(len(q)),4))

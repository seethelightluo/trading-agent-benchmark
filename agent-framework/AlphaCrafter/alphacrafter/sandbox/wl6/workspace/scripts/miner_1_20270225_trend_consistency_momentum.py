import pandas as pd, numpy as np, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-02-24')
P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
px=pd.DataFrame(P); R=px.pct_change()
mean20=R.rolling(20,min_periods=15).mean(); absmean20=R.abs().rolling(20,min_periods=15).mean()
cons=(mean20/absmean20).shift(1); ret20=px.pct_change(20).shift(1); vol=R.rolling(20,min_periods=15).std().shift(1)
f=(ret20/vol*cons).replace([np.inf,-np.inf],np.nan)
rows=[]
for a in A:
 for h in [1,3,5,10]:
  fr=px[a].shift(-h)/px[a]-1
  for dt in f.index: rows.append((dt,a,h,f.loc[dt,a],fr.loc[dt]))
dd=pd.DataFrame(rows,columns=['date','a','h','f','r']).dropna()
for h in [1,3,5,10]:
 obs=[]; ns=[]
 for dt,g in dd[dd.h==h].groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   z=spearmanr(g.f,g.r).statistic
   if np.isfinite(z): obs.append(z); ns.append(len(g))
 x=np.array(obs); print('H',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),5),'ICIR',round(x.mean()/x.std(ddof=1),5),'hit',round(np.mean(x>0),4))
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
   q=[]
   for dt,g in dd[(dd.h==1)&(dd.date.dt.year>=lo)&(dd.date.dt.year<=hi)].groupby('date'):
    if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:q.append(spearmanr(g.f,g.r).statistic)
   q=np.array(q); print('REG',lo,hi,'dates',len(q),'IC',round(q.mean(),5) if len(q) else np.nan,'ICIR',round(q.mean()/q.std(ddof=1),5) if len(q)>1 else np.nan)
print('UNIVERSE',len(A),'date_start',dd.date.min(),'date_end',dd.date.max())

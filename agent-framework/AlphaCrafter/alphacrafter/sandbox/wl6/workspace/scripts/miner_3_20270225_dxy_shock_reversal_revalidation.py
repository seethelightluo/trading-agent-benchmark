import pandas as pd, numpy as np, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-02-24')
P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
R=pd.DataFrame(P).pct_change(); m=pd.read_csv('../persistent/index_data/DXY.csv'); m.date=pd.to_datetime(m.date); d=m.sort_values('date').set_index('date').close.loc[:cut]; mr=d.pct_change(); sd=mr.rolling(60,min_periods=40).std().shift(1); shock=(mr.shift(1)/sd).clip(-4,4); act=((shock.abs()-.75).clip(lower=0)/1.75).clip(upper=1)
rows=[]
for a in A:
 vol=R[a].rolling(20,min_periods=15).std().shift(1); f=(-R[a].shift(1)/vol*act.reindex(R.index)).replace([np.inf,-np.inf],np.nan)
 for h in [1,5,10]:
  fr=P[a].shift(-h)/P[a]-1
  for dt in f.index: rows.append((dt,a,h,f.loc[dt],fr.reindex(f.index).loc[dt]))
dd=pd.DataFrame(rows,columns=['date','a','h','f','r']).dropna(); print('cut',cut.date(),'assets',len(A),'rows',len(dd))
for h in [1,5,10]:
 obs=[]; ns=[]
 for dt,g in dd[dd.h==h].groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   x=spearmanr(g.f,g.r).statistic
   if np.isfinite(x): obs.append(x); ns.append(len(g))
 x=np.array(obs); print('H',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),5),'ICIR',round(x.mean()/x.std(ddof=1),5),'hit',round(np.mean(x>0),4))
 if h==1:
  for lo,hi in [(2025,2026),(2027,2027)]:
   q=[]
   for dt,g in dd[(dd.h==1)&(dd.date.dt.year>=lo)&(dd.date.dt.year<=hi)].groupby('date'):
    if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:q.append(spearmanr(g.f,g.r).statistic)
   q=np.array(q); print('REG',lo,hi,len(q),round(q.mean(),5) if len(q) else np.nan,round(q.mean()/q.std(ddof=1),5) if len(q)>1 else np.nan)

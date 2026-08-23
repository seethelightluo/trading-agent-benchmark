import pandas as pd, numpy as np, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-01-27'); base='../persistent/stock_data'; macro='../persistent/index_data'
px={}
for a in A:
 d=pd.read_csv(f'{base}/{a}.csv'); d.date=pd.to_datetime(d.date); px[a]=d.sort_values('date').set_index('date').close.loc[:cut]
P=pd.DataFrame(px); R=P.pct_change()
m=pd.read_csv(f'{macro}/DXY.csv'); m.date=pd.to_datetime(m.date); dxy=m.sort_values('date').set_index('date').close.loc[:cut]
mr=dxy.pct_change(); sd=mr.rolling(60,min_periods=40).std().shift(1)
shock=(mr.shift(1)/sd).clip(-4,4)
# smooth positive/negative dollar shock; reversal is contrarian to prior asset return
act=((shock.abs()-0.75).clip(lower=0)/1.75).clip(upper=1)
sig={}
for a in A:
 vol=R[a].rolling(20,min_periods=15).std().shift(1)
 sig[a]=(-R[a].shift(1)/vol*act).replace([np.inf,-np.inf],np.nan)
rows=[]
for a in A:
 for h in [1,3,5,10]:
  fr=P[a].shift(-h)/P[a]-1
  for dt in sig[a].index: rows.append((dt,a,h,sig[a].loc[dt],fr.loc[dt]))
dd=pd.DataFrame(rows,columns=['date','a','h','f','r']).dropna()
for h in [1,3,5,10]:
 z=dd[dd.h==h]; obs=[]; ns=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
   v=spearmanr(g.f,g.r).statistic
   if np.isfinite(v): obs.append(v); ns.append(len(g))
 obs=np.array(obs); print('H',h,'dates',len(obs),'avg_n',round(np.mean(ns),2),'IC',round(obs.mean(),5),'ICIR',round(obs.mean()/obs.std(ddof=1),5),'hit',round((obs>0).mean(),4))
z=dd[dd.h==1]; obs=[]
for dt,g in z.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1: obs.append((dt,spearmanr(g.f,g.r).statistic))
z=pd.DataFrame(obs,columns=['date','ic'])
print('PERIOD',z.date.min().date(),z.date.max().date())
for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
 q=z[(z.date.dt.year>=lo)&(z.date.dt.year<=hi)]; print('REG',lo,hi,len(q),round(q.ic.mean(),5),round(q.ic.mean()/q.ic.std(ddof=1),5) if len(q)>1 else np.nan)

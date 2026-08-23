import pandas as pd,numpy as np,warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-02-24')
P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);P[a]=d.sort_values('date').set_index('date').close.loc[:cut]
px=pd.DataFrame(P);R=px.pct_change(); csdisp=R.rolling(5,min_periods=4).std().mean(axis=1); base=csdisp.rolling(60,min_periods=40).rank(pct=True).shift(1)
f=(-px.pct_change(5).div(R.rolling(20,min_periods=15).std()).mul((base-.5).clip(lower=0)*2)).shift(1)
rows=[]
for a in A:
 fr=px[a].shift(-1)/px[a]-1
 for dt in f.index: rows.append((dt,a,f.loc[dt,a],fr.loc[dt]))
dd=pd.DataFrame(rows,columns=['date','a','f','r']).dropna();obs=[];ns=[]
for dt,g in dd.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:
  z=spearmanr(g.f,g.r).statistic
  if np.isfinite(z):obs.append(z);ns.append(len(g))
x=np.array(obs);print('dates',len(x),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),5),'ICIR',round(x.mean()/x.std(ddof=1),5),'hit',round(np.mean(x>0),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
 q=[]
 for dt,g in dd[(dd.date.dt.year>=lo)&(dd.date.dt.year<=hi)].groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:q.append(spearmanr(g.f,g.r).statistic)
 q=np.array(q);print('REG',lo,hi,len(q),round(q.mean(),5) if len(q) else np.nan,round(q.mean()/q.std(ddof=1),5) if len(q)>1 else np.nan)

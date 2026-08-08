import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(f)[:-4]
 if s in keep:
  x=pd.read_csv(f); x.date=pd.to_datetime(x.date); d[s]=x.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2033-03-16']; r=px.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv');vix.date=pd.to_datetime(vix.date); v=vix.set_index('date').close.reindex(px.index).ffill(); vr=v.pct_change()
# VIX-up resilience: mean asset return on recent VIX-up sessions, risk-normalized and lagged.
up=vr>0
num=r.where(up, np.nan).rolling(40,min_periods=10).mean()
vol=r.rolling(20,min_periods=15).std()
f=(num/(vol+1e-12)).shift(1)
print('candidate vix_up_resilience_40_riskadj; dates',len(px),'assets',len(px.columns))
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1;a=[];ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 a=np.array(a);print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for label,a,b in [('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-33','2031','2033-03-16')]:
 fr=px.shift(-1)/px-1;z=[]
 for dt in f.loc[a:b].index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=np.array(z);print('REG1',label,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('coverage',round(f.notna().mean().mean(),4),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4))

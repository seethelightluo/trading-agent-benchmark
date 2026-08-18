import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s,macro=False):
 p=('../persistent/index_data/' if macro else '../persistent/stock_data/')+s+'.csv'; d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d.set_index('date').close
px=pd.concat({s:load(s) for s in U},axis=1).sort_index();r=px.pct_change()
v=load('VIX',1).reindex(px.index).ffill(); vr=v.pct_change(); vz=(vr-vr.rolling(60,min_periods=40).mean())/vr.rolling(60,min_periods=40).std()
b=pd.DataFrame({s:r[s].rolling(60,min_periods=40).cov(vr)/(vr.rolling(60,min_periods=40).var()+1e-8) for s in U})
# Stress-only defensive exposure: reward low VIX beta when VIX shock is elevated.
stress=np.maximum(vz,0)
fac=-b*stress.values[:,None];fac=fac.sub(fac.mean(axis=1),axis=0)
for h in [1,5,10]:
 a=[];n=[]
 for i in range(len(px)-h):
  q=pd.concat([fac.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);n.append(len(q))
 a=np.array(a);print('stress_beta',h,'dates',len(a),'avgN',round(np.mean(n),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
print('turnover',round(fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),5))
for years in [(2020,2022),(2023,2024),(2025,2026)]:
 a=[]
 for i in range(len(px)-1):
  if years[0]<=px.index[i].year<=years[1]:
   q=pd.concat([fac.iloc[i],px.iloc[i+1]/px.iloc[i]-1],axis=1).dropna()
   if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.array(a);print('regime',years,'dates',len(a),'ICIR',round(a.mean()/a.std(),5))

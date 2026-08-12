import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv');d.date=pd.to_datetime(d.date);p[a]=d.sort_values('date').set_index('date').close.astype(float)
pd_=pd.DataFrame(p).sort_index(); r=pd_.pct_change()
# downside asymmetry / consistency: upside-day share minus downside magnitude share, lagged
pos=(r>0).rolling(20,min_periods=15).mean(); down=(-r.clip(upper=0)).rolling(20,min_periods=15).mean(); up=r.clip(lower=0).rolling(20,min_periods=15).mean()
f=((pos-0.5)*(up+down)/(r.rolling(20,min_periods=15).std()+1e-12)).shift(1)
for h in [1,3,5,10]:
 y=pd_.pct_change(h).shift(-h); out=[]
 for dt in pd_.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:out.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(out,columns=['d','n','ic']).set_index('d');x=q.ic
 print('h',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
 for nm,s in [('2020-22',q.loc['2020':'2022']),('2023-25',q.loc['2023':'2025']),('2026-27',q.loc['2026':'2027']),('2028',q.loc['2028':])]:
  z=s.ic
  print(nm,len(z),round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('coverage',round(f.notna().sum().sum()/(len(pd_)*len(A)),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))

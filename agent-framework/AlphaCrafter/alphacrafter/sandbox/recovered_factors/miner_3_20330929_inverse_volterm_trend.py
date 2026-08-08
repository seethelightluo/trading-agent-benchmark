import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d.date=pd.to_datetime(d.date); px[a]=d.set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
vs=r.rolling(5,min_periods=4).std()/r.rolling(60,min_periods=40).std()
trend=p.pct_change(20)/(r.rolling(20,min_periods=15).std()+1e-12)
# Pre-specified inverse orientation: continuation is favored when short vol is elevated
sig=(-trend/(vs+0.05)).shift(1)
print('range',p.index.min().date(),p.index.max().date(),'assets',len(p.columns),'validcells',int(sig.notna().sum().sum()))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z);print('H',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 f=p.shift(-10)/p-1;z=[]
 for dt in sig.loc[lo:hi].index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=np.array(z);print('REG',lo+'-'+hi,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
print('coverage',round(sig.notna().mean().mean(),4),'meanvalid',round(sig.notna().sum(axis=1).mean(),2),'turn10',round(sig.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))

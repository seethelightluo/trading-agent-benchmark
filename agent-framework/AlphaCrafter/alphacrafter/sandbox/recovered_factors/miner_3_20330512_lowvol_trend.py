import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x.date=pd.to_datetime(x.date); P[a]=x.set_index('date').close
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); ret=p.pct_change(30)
# low-vol trend: 30d return, attenuated by own volatility relative to its trailing 120d vol; lag one day
vr=vol/vol.rolling(120,min_periods=60).median(); sig=(ret/(1+vr)).shift(1)
print('range',p.index.min(),p.index.max(),'assets',len(assets))
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; ic=[]; ns=[]
 for d in p.index:
  z=pd.concat([sig.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=np.asarray(ic); print('H',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),3))
rank=sig.rank(axis=1,pct=True); print('coverage',sig.notna().sum().mean()/15,'turnover10',((rank-rank.shift(10)).abs().mean(axis=1)).mean(),'mean_valid',sig.notna().sum(axis=1).mean())
for y in sorted(p.index.year.unique()):
 x=[]; fw=p.shift(-1)/p-1
 for d in p.index[p.index.year==y]:
  z=pd.concat([sig.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 if len(x)>20: print('YEAR',y,len(x),round(np.mean(x),5),round(np.mean(x)/np.std(x),4))
print('library audit deferred: efficacy required before admitted-factor correlation; candidate cells',int(sig.notna().sum().sum()))

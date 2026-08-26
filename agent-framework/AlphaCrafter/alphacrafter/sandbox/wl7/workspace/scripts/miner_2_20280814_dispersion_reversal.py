import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-08-14')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); P[s]=d[d.index<=end].close
p=pd.DataFrame(P); r=p.pct_change(); ret5=p.shift(1)/p.shift(6)-1
# dispersion is observable at t-1; suppress reversal when dispersion is low, and normalize by asset vol
vol=r.shift(1).rolling(20,min_periods=15).std(); disp=ret5.std(axis=1)
f=-ret5/(vol+1e-8)*disp.to_numpy()[:,None]
print('dispersion-weighted 5d reversal')
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; z=[];ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic);ns.append(len(a))
 z=np.array(z); rr=z[-252:]
 print('h',h,'IC %.6f ICIR %.4f dates %d avgN %.2f recentIC %.6f recentICIR %.4f'%(z.mean(),z.mean()/z.std(ddof=1),len(z),np.mean(ns),rr.mean(),rr.mean()/rr.std(ddof=1)))
print('coverage',f.notna().sum().sum()/f.size,'rank turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-08-14')]:
 z=[];fr=p.shift(-10)/p-1
 for dt in f.loc[a:b].index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 z=np.array(z);print('regime',a,'n',len(z),'IC10',z.mean(),'ICIR',z.mean()/z.std(ddof=1))

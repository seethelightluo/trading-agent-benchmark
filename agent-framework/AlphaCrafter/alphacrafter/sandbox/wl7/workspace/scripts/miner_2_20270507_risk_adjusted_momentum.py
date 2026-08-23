import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f); x.date=pd.to_datetime(x.date); P[s]=x.set_index('date').sort_index().close.astype(float)
p=pd.DataFrame(P).sort_index(); r=p.pct_change()
# lagged 20-day risk-adjusted momentum, with cross-sectional neutralization implicit in ranks
sig=p.shift(1).pct_change(20)/(r.shift(1).rolling(20).std()*np.sqrt(20))
ff=p.shift(-1)/p-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15));print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean());print('turnover',sig.rank(axis=1).diff().abs().mean(axis=1).mean()/15)
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 x=q.loc[a:b];print(a,len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1) if len(x)>1 else np.nan)
for h in [5,10,20]:
 f=p.shift(-h)/p-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(rr),len(rr))
sig.to_csv('scripts/miner_2_20270507_risk_adjusted_momentum_signal.csv');q.to_csv('scripts/miner_2_20270507_risk_adjusted_momentum_ic.csv')

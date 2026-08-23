import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2032-07-21'; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]; r=x.close.pct_change()
 # directionally persistent trend: net 60d return divided by absolute daily path
 x['f']=x.close.pct_change(60)/(r.abs().rolling(60,min_periods=50).sum()+1e-12)
 x['y']=x.close.shift(-20)/x.close-1; D[s]=x[['f','y']]
dates=sorted(set.intersection(*[set(x.index) for x in D.values()])); a=[];ns=[]; used=[]
for dt in dates:
 z=pd.DataFrame({s:[D[s].loc[dt,'f'],D[s].loc[dt,'y']] for s in U},index=['f','y']).T.dropna()
 if len(z)>=8:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));used.append(dt)
a=np.array(a); print('dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR_daily',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'coverage',np.mean(ns)/15)
for y in range(2026,2033):
 q=a[[d.year==y for d in used]]; print(y,np.mean(q),len(q))

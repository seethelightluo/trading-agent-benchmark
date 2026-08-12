import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
dates=sorted(set.intersection(*[set(x.index) for x in D.values()]));C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U});R=C.pct_change()
# Magnitude-weighted directional efficiency: trend efficiency multiplied by signed 20d return; lagged.
eff=(C/C.shift(20)-1)/(R.abs().rolling(20,min_periods=18).sum()+1e-12); F=(eff*(C/C.shift(20)-1)).shift(1);Y=C.shift(-1)/C-1
A=[];ns=[];ds=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:A.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
a=np.array(A);print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',F.notna().sum().sum()/F.size,'turn',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for yr in range(2020,2027):
 q=np.array([v for d,v in zip(ds,a) if d.year==yr]);
 if len(q):print('regime',yr,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [5,10]:
 Y=C.shift(-h)/C-1;a=[]
 for dt in dates:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:a.append(spearmanr(z.f,z.y).statistic)
 a=np.array(a);print('h',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1))

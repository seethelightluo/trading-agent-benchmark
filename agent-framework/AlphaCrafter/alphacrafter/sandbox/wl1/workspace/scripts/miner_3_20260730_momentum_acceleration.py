import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
p=pd.concat({s:load(s).close for s in U},axis=1).sort_index(); r=p.pct_change(); m=r.median(axis=1)
def residual(window):
 out=pd.DataFrame(index=p.index,columns=U,dtype=float)
 for i in range(window,len(p)):
  h=r.iloc[i-window:i]; mm=m.iloc[i-window:i]; v=mm.var(); beta=h.apply(lambda z:z.cov(mm)/v if v>1e-12 else 0)
  out.iloc[i]=(p.iloc[i]/p.iloc[i-window]-1)-beta*(np.prod(1+mm)-1)
 return out
# Acceleration: short residual momentum minus long residual momentum, interpretable trend change signal
f=residual(20)-residual(60)
for horizon in [1,5,10,20]:
 ic=[]; ns=[]; dates=[]
 for i in range(60,len(p)-horizon):
  q=pd.concat([f.iloc[i],(p.iloc[i+horizon]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   ic.append(spearmanr(q.iloc[:,0],q.y).statistic); ns.append(len(q)); dates.append(p.index[i])
 a=np.array(ic); print('horizon',horizon,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if horizon==10: print('regime',{int(y):round(a[np.array([d.year for d in dates])==y].mean(),5) for y in sorted(set(d.year for d in dates))})
# simple signal turnover using rank direction changes
print('assets',len(U),'date_range',p.index[0].date(),p.index[-1].date())

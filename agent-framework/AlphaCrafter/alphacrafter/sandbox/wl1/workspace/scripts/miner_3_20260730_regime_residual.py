import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def ld(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut].close
p=pd.concat({s:ld(s) for s in U},axis=1).sort_index(); r=p.pct_change(); med=r.median(axis=1)
# regime-conditioned residual momentum: residualize 20d returns to cross-asset median, then use trend in positive breadth and fade it in stressed breadth
w=20; f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i in range(60,len(p)):
 h=r.iloc[i-w:i]; mm=med.iloc[i-w:i]; v=mm.var(); beta=h.apply(lambda z:z.cov(mm)/v if v>1e-12 else 0)
 base=(p.iloc[i]/p.iloc[i-w]-1)-beta*(np.prod(1+mm)-1)
 breadth=(r.iloc[i-10:i]>0).mean().mean()
 f.iloc[i]=base if breadth>=.5 else -base
for hor in [1,5,10,20]:
 a=[]; ns=[]; ds=[]
 for i in range(60,len(p)-hor):
  q=pd.concat([f.iloc[i],(p.iloc[i+hor]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:a.append(spearmanr(q.iloc[:,0],q.y).statistic);ns.append(len(q));ds.append(p.index[i])
 a=np.array(a); print('horizon',hor,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 if hor==10:print('regime',{int(y):round(a[np.array([d.year for d in ds])==y].mean(),5) for y in sorted(set(d.year for d in ds))})
print('assets',len(U),'range',p.index[0].date(),p.index[-1].date())

import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-11-18')
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}
p=pd.DataFrame(p).sort_index(); r=p.pct_change(); m=r.median(axis=1)
# rolling market-beta residual: remove each asset's 20d beta to cross-sectional median, then reverse 3d residual
for W,L in [(20,3),(40,3),(20,5)]:
 cov=r.rolling(W,min_periods=W).cov(m); var=m.rolling(W,min_periods=W).var()
 beta=cov.div(var,axis=0); resid=r-beta.mul(m,axis=0)
 f=-resid.rolling(L,min_periods=L).sum()
 for h in [1,5,10]:
  q=[]; dates=[]; counts=[]
  for i in range(len(p)-h):
   z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
    q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(p.index[i]); counts.append(len(z))
  q=np.asarray(q); print('W',W,'L',L,'h',h,'dates',len(q),'avgN',round(np.mean(counts),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round(np.mean(q>0),4))
 print('coverage',round(f.notna().sum().sum()/f.size,4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
print('range',p.index.min(),p.index.max(),'rows',len(p))

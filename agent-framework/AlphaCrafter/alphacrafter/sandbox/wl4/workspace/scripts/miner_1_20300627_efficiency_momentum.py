import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'
px={}
for s in U:
 f=os.path.join(root,s+'.csv')
 d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
 px[s]=d['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# interpretable range-efficiency momentum: signed net move / total path, with mild volatility scaling
# value at t uses through t; cross-sectional signal at t is shifted one day before forward return
for horizon in [10,20]:
 rows=[]
 for i in range(30,len(p)-horizon):
  t=p.index[i]
  # efficiency of recent path, retaining direction and normalized by realized vol
  ret=p.iloc[i]/p.iloc[i-20]-1
  path=r.iloc[i-19:i+1].abs().sum()
  sig=(ret/path.replace(0,np.nan)).shift(0)
  fwd=p.iloc[i+horizon]/p.iloc[i]-1
  z=pd.concat([sig,fwd],axis=1).dropna()
  if len(z)>=8:
   rows.append((t,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 for label,y in [('full',x),('recent',x.tail(261)),('regime1',x.iloc[:len(x)//3]),('regime2',x.iloc[len(x)//3:2*len(x)//3]),('regime3',x.iloc[2*len(x)//3:])]:
  ic=y.ic.mean(); sd=y.ic.std(ddof=1); ir=ic/sd*np.sqrt(len(y)) if sd else np.nan
  print(horizon,label,'dates',len(y),'avgN',round(y.n.mean(),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((y.ic>0).mean(),4))
print('coverage',p.notna().mean().mean(),'assets',len(U),'last',p.index[-1].date())

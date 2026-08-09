import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date'); return d.close
p=pd.concat({s:ld(s) for s in U},axis=1).sort_index(); r=p.pct_change()
# defensive basket return, equal weighted XAU, US10Y, CN10Y; factor rewards positive sensitivity to defense
b=r[['XAU','US10Y','CN10Y']].mean(axis=1)
f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for s in U:f[s]=r[s].rolling(60,min_periods=45).cov(b)/b.rolling(60,min_periods=45).var()
# defensive beta itself: higher beta should lead in risk-off basket
for h in [1,5,10]:
 z=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z);print(h,len(z),np.mean(ns),z.mean(),z.mean()/z.std(),(z>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',((f.rank(axis=1)-f.rank(axis=1).shift()).abs().mean().mean()/15))
print('regimes',[(y,np.nanmean([z for i,z in enumerate([])])) for y in []])
# correlation with momentum and reversal proxies
mom=r.rolling(20).sum(); rev=-r.rolling(5).sum(); print('corr',f.stack().corr(mom.stack()),f.stack().corr(rev.stack()))

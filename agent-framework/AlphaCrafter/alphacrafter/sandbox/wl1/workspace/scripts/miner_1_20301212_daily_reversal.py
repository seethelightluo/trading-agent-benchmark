import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(f); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# one-day reversal, lagged one completed session (factor at t uses return through t-1)
f=(-r.shift(1)).replace([np.inf,-np.inf],np.nan)
# evaluate forward close-to-close returns at horizons
for h in [1,2,5,10,20]:
 fr=p.shift(-h)/p-1
 vals=[]; dates=[]
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));dates.append(dt)
 ic=pd.Series(vals,index=dates).dropna()
 print(h,'dates',len(ic),'avgN',round(np.mean([pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna().shape[0] for d in dates]),2),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit', (ic>0).mean())
# coverage and rank turnover
print('coverage',f.notna().mean().mean(),'obs',f.notna().sum().sum())
q=f.rank(axis=1,pct=True); turn=q.diff().abs().mean(axis=1).mean(); print('turnover',turn)
print('period',p.index.min(),p.index.max())
# regime annual daily
fr=p.shift(-1)/p-1
for yr,g in pd.DataFrame({'f':f.stack(),'y':fr.stack()}).dropna().groupby(pd.Grouper(level=0,freq='YE')):
 cs=[]
 for d,x in g.groupby(level=0):
  if len(x)>=8: cs.append(x.f.corr(x.y,method='spearman'))
 if cs: print(str(yr.date()),len(cs),np.mean(cs),np.mean(cs)/np.std(cs,ddof=1))

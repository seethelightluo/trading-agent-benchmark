import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 D[a]=x
idx=sorted(set.intersection(*[set(x.index) for x in D.values()]))
rows=[]
for t in idx:
 vals=[]; fut=[]
 for a in assets:
  x=D[a]; k=x.index.get_loc(t)
  if k<2 or k+1>=len(x): continue
  prev=x.iloc[k-1]; cur=x.iloc[k]; nxt=x.iloc[k+1]
  gap=cur.open/prev.close-1; fr=nxt.close/cur.close-1
  if np.isfinite(gap) and np.isfinite(fr): vals.append(-gap); fut.append(fr)
 if len(vals)>=8: rows.append((t,spearmanr(vals,fut).statistic,len(vals)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'assets',len(assets),'meanN',r.n.mean(),'coverage cells',r.n.sum()/(len(idx)*15))
z=r.ic.dropna(); print('IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030')]:
 z=r.loc[lo:hi].ic; print(lo,len(z),z.mean(),z.mean()/z.std(ddof=1))
z=r.tail(120).ic; print('latest120',len(z),z.mean(),z.mean()/z.std(ddof=1))
print('library_files',len(glob.glob('factors/*.json')))

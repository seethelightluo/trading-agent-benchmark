import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; px={}
for s in S:
 f=f'{base}/{s}.csv'
 if os.path.exists(f): px[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
p=pd.DataFrame(px).sort_index(); p=p.loc[:'2027-09-22']; r=p.pct_change(); vol=r.rolling(20,min_periods=15).std();
# Short-horizon reversal, normalized by recent risk and mildly smoothed; signal uses close through t only
f=(-r.rolling(3,min_periods=3).sum()/vol.replace(0,np.nan)).ewm(span=3,min_periods=3,adjust=False).mean().clip(-8,8)
rows=[]; y=r.shift(-1)
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/(len(o)*15)); print('IC %.8f ICIR %.8f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean())); print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020-01-01','2021-12-31'),('2022-01-01','2023-12-31'),('2024-01-01','2025-12-31'),('2026-01-01','2027-12-31')]:
 q=o.loc[a:b].ic; print('regime',a,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [3,5,10]:
 yy=(1+r).rolling(h).apply(np.prod,raw=True).shift(-h); zics=[]
 for d in f.index:
  z=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8:zics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(zics); print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'dates',len(q))
print('max_abs_library_correlation unavailable')

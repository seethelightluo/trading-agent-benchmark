import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try: x=get_stock_daily_data(s,days=4000)
 except: x=None
 if x is not None and len(x)>100:D[s]=x.set_index(pd.to_datetime(x.date)).close.astype(float).sort_index()
p=pd.DataFrame(D).sort_index().loc[:'2029-06-13'];r=p.pct_change();f=(p/p.shift(40)-1)/(r.rolling(40).std()*np.sqrt(40))*((r>0).rolling(40).mean()-.5)
for H in [1,5,10,20]:
 fr=p.shift(-H)/p-1; q=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 q=pd.Series(q).dropna();print(f'H={H} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.4f}')
print(f'coverage={(f.notna().sum(axis=1)/len(U)).mean():.6f} rank_turnover_proxy={f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean():.6f} assets={len(D)} dates={len(p)}')

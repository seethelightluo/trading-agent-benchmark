import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
P=pd.DataFrame(p); r=P.pct_change();
# residualized 3-day reversal, conditioned on broad cross-asset move
x=-r.rolling(3).sum(); f=x.sub(x.median(axis=1),axis=0)
# orthogonalize against 1-day reversal cross-section each date
one=-r; beta=[]
for dt in P.index:
 z=pd.concat([f.loc[dt],one.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,1].var()>0:
  b=np.cov(z.iloc[:,0],z.iloc[:,1],ddof=1)[0,1]/z.iloc[:,1].var(); q=z.iloc[:,0]-b*z.iloc[:,1]
  beta.append((dt,q.to_dict()))
rows=[]; sig=[]
for dt,q in beta:
 for a,v in q.items(): sig.append((dt,a,v))
 nxt=P.shift(-1).loc[dt]
 z=pd.DataFrame({'f':pd.Series(q),'y':nxt/P.loc[dt]-1}).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']);
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=d.set_index('date').loc[lo:hi].ic;print(lo,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('ALL dates',len(d),'avgN',d.n.mean(),'IC',d.ic.mean(),'ICIR',d.ic.mean()/d.ic.std(ddof=1),'hit',(d.ic>0).mean())
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_1_20270225_resid_3d_1d.csv',index=False);print('artifact',len(out),'coverage',out.signal.notna().mean(),'turnover',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())

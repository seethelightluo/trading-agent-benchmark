import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
P=pd.DataFrame(p); r=P.pct_change(); med=r.median(axis=1)
# peer spillover: 5-day peer median return, orthogonalized to own 20-day momentum
peer=r.sub(r,axis=0) # placeholder
peer=pd.DataFrame({a:r.drop(columns=a).median(axis=1) for a in A})
spill=peer.rolling(5,min_periods=5).sum()
ownmom=r.rolling(20,min_periods=20).sum()
# cross-sectional residual: remove linear own momentum component each date
raw=pd.DataFrame(index=P.index,columns=A,dtype=float)
for dt in P.index:
 x=pd.concat([spill.loc[dt].rename('spill'),ownmom.loc[dt].rename('mom')],axis=1).dropna()
 if len(x)>=8 and x.mom.var()>1e-12:
  b=np.cov(x.spill,x.mom,ddof=1)[0,1]/x.mom.var(); raw.loc[dt,x.index]=x.spill-b*x.mom
rows=[]; sig=[]
for dt in P.index:
 vals=raw.loc[dt];
 for a in A: sig.append((dt,a,vals[a]))
 for h in [1,3,5,10]:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; q=pd.concat([vals.rename('f'),y.rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: rows.append((dt,h,spearmanr(q.f,q.y).statistic,len(q)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10]:
 q=d[d.h==h]; print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  x=q.set_index('date').loc[lo:hi].ic; print(lo,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']); out.to_csv('../persistent/factor_signals_miner_2_20270227_resid_peer_spillover.csv',index=False)
w=out.pivot(index='date',columns='asset',values='signal'); print('artifact',len(out),'coverage',round(out.signal.notna().mean(),4),'turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))

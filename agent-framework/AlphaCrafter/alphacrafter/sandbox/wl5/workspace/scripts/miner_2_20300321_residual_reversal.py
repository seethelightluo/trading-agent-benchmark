import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/index_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): D[s]=pd.read_csv(f,parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Residual medium-horizon reversal: remove common cross-asset movement, scale by own volatility.
cs=r.mean(axis=1)
beta=r.rolling(60).cov(cs).div(cs.rolling(60).var(),axis=0)
res=r.sub(beta.mul(cs,axis=0),axis=0)
res20=res.rolling(20).sum(); vol20=r.rolling(20).std()*np.sqrt(252)
sig=-(res20/vol20)
h=10; fwd=p.shift(-h)/p-1; vals=[]; dates=[]; ns=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): vals.append(c); dates.append(dt); ns.append(len(z))
q=pd.Series(vals,index=dates); ic=q.mean(); icir=ic/q.std()
print('horizon',h,'dates',len(q),'avg_instruments',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f'%(ic,icir,(q>0).mean()))
for label,lo,hi in [('2020-24','2020','2025'),('2025-27','2025','2028'),('2028-30','2028','2030-03-22')]:
 t=q[(q.index>=lo)&(q.index<hi)]; print(label,'dates',len(t),'IC %.8f ICIR %.8f hit %.4f'%(t.mean(),t.mean()/t.std(),(t>0).mean()))
print('coverage %.6f turnover %.6f instruments %d rows %d'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean().mean(),len(D),int(sig.stack().size)))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20300321_residual_reversal_signal.csv',index=False)
print('artifact',len(out))

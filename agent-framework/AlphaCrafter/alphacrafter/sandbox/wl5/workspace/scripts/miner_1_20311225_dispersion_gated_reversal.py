import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 try:d=get_stock_daily_data(s,days=4000)
 except Exception:d=None
 if d is not None and len(d):px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill();r=P.pct_change();rel5=r.rolling(5).sum().sub(r.rolling(5).sum().mean(axis=1),axis=0);vol20=r.rolling(20).std()*np.sqrt(252);disp20=r.rolling(20).std().mean(axis=1);gate=(disp20/disp20.rolling(120).median()).clip(.5,2);f=(-rel5/vol20)*gate.values[:,None];fr=P.pct_change(10).shift(-10)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
D=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');ic=D.ic.mean();icir=ic/D.ic.std(ddof=1);S=f.div(f.abs().sum(axis=1),axis=0);to=S.diff().abs().sum(axis=1).dropna().reindex(D.index).mean()
print('candidate=dispersion_gated_relative_reversal_10d');print('assets',len(px),'dates',len(D),'mean_instruments',D.n.mean(),'coverage_pct',D.n.mean()/len(U)*100);print('IC',ic,'ICIR',icir,'hit',float((D.ic>0).mean()),'turnover',to)
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-12-24')]:
 q=D.loc[a:b].ic;print(a,b,'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10,20]:
 yy=P.pct_change(h).shift(-h);q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q,ddof=1))
out=f.stack().rename('signal').to_frame();out.index.names=['date','symbol'];out.to_csv('scripts/miner_1_20311225_dispersion_gated_reversal_signal.csv')

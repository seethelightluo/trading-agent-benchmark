import os,numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2028-11-29'); P={}
for s in U:
 d=get_stock_daily_data(s,4000); d=d[d.date<=cutoff].copy(); P[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); m=r.mean(axis=1); wm=m.rolling(120,min_periods=80).mean(); vr=((m-wm)**2).rolling(120,min_periods=80).mean()
res=pd.DataFrame(index=r.index,columns=r.columns,dtype=float)
for c in r:
 wr=r[c].rolling(120,min_periods=80).mean(); cov=((r[c]-wr).mul(m-wm)).rolling(120,min_periods=80).mean(); res[c]=r[c]-cov/vr*m
sig=res.rolling(40,min_periods=35).sum().shift(1); fwd=px.shift(-20)/px-1; rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); turn=sig.rank(pct=True).diff().abs().mean(axis=1).mean(); print('assets',len(px.columns),'dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15); print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(),'hit',(r.ic>0).mean(),'turnover',turn)
for lab,st in [('2020+','2020'),('2026+','2026'),('2027+','2027'),('2028+','2028')]:
 q=r[r.index>=st]; print(lab,len(q),q.ic.mean(),q.ic.mean()/q.ic.std())
os.makedirs('scripts',exist_ok=True); sig.to_csv('scripts/miner_3_20281130_equal_residual_momentum_signal.csv',index_label='date')

import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for s in U:
 d=get_stock_daily_data(s,days=2500)
 if d is not None and len(d)>150: d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in raw.items()}); op=pd.DataFrame({s:d.open for s,d in raw.items()}); r=cl.pct_change()
intra=(cl/op-1).replace([np.inf,-np.inf],np.nan)
vol=r.rolling(20,min_periods=12).std().shift(1)
pressure=intra.rolling(5,min_periods=4).sum().shift(1)
fac=(-pressure/vol).clip(-10,10); fac=fac.sub(fac.median(axis=1),axis=0)
rows=[]
for dt in fac.index:
 ix=cl.index.get_loc(dt); v=fac.loc[dt]
 if v.notna().sum()<8: continue
 for h in [1,3,5,10]:
  if ix+h>=len(cl): continue
  z=pd.concat([v.rename('f'),(cl.iloc[ix+h]/cl.iloc[ix]-1).rename('r')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),z.f.corr(z.r,method='spearman')))
w=pd.DataFrame(rows,columns=['date','h','n','ic']); print('assets',len(raw),'dates',len(cl),'range',cl.index.min(),cl.index.max())
for h in [1,3,5,10]:
 x=w[w.h==h]; q=x.ic; print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f coverage %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),x.n.mean(),x.n.mean()/len(U)))
 for lab,n in [('recent180',180),('recent360',360)]:
  y=q.tail(n); print(lab,'%.6f %.6f'%(y.mean(),y.mean()/y.std(ddof=1)))
rank=fac.rank(axis=1,pct=True); t=[]
for i in range(1,len(rank)):
 z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-z.iloc[:,0].corr(z.iloc[:,1]))
print('turnover_proxy',np.mean(t),'coverage',fac.notna().sum(axis=1).mean()/len(U))
out=[{'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(fac.loc[d,s])} for d in fac.index for s in fac.columns if pd.notna(fac.loc[d,s])]; pd.DataFrame(out).to_csv('scripts/miner_2_20290726_intraday_volnorm_reversal_signal.csv',index=False); print('artifact_rows',len(out))

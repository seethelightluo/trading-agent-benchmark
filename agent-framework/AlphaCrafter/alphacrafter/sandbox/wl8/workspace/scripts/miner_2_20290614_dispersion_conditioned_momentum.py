import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=2500)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in raw.items()}); r=cl.pct_change()
# Lagged 20d return, demeaned cross-section, scaled by lagged cross-sectional dispersion.
# The nonlinear state multiplier emphasizes continuation when dispersion is elevated,
# while retaining interpretable direction and avoiding future information.
mom=r.rolling(20,min_periods=10).sum().shift(1)
csdisp=r.rolling(20,min_periods=10).std().shift(1).mean(axis=1)
state=(csdisp/csdisp.rolling(120,min_periods=40).median()).clip(0.5,2.0)
fac=mom.sub(mom.median(axis=1),axis=0).mul(state,axis=0)
rows=[]
for dt in fac.index:
 vals=fac.loc[dt]; ix=cl.index.get_loc(dt)
 if vals.notna().sum()<8: continue
 for h in [1,3,5,10]:
  if ix+h>=len(cl.index): continue
  z=pd.concat([vals.rename('f'),(cl.iloc[ix+h]/cl.iloc[ix]-1).rename('r')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),z.f.corr(z.r,method='spearman')))
rw=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('assets',len(raw),'dates',len(cl),'range',cl.index.min(),cl.index.max())
for h in [1,3,5,10]:
 x=rw[rw.h==h].dropna(); ic=x.ic
 print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f coverage %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),len(ic),x.n.mean(),x.n.mean()/len(U)))
 for label,cut in [('recent180',180),('recent360',360)]:
  y=ic.tail(cut); print(label,'%.6f %.6f'%(y.mean(),y.mean()/y.std(ddof=1)))
rank=fac.rank(axis=1,pct=True); rc=[]
for i in range(1,len(rank)):
 z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(z)>=8: rc.append(1-z.iloc[:,0].corr(z.iloc[:,1]))
print('turnover_proxy',np.nanmean(rc),'coverage',fac.notna().sum(axis=1).mean()/len(U))
out=[{'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(fac.loc[dt,s])} for dt in fac.index for s in fac.columns if pd.notna(fac.loc[dt,s])]
pd.DataFrame(out).to_csv('scripts/miner_2_20290614_dispersion_conditioned_momentum_signal.csv',index=False)
print('artifact_rows',len(out))

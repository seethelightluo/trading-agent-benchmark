import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=2500)
 if d is not None and len(d)>100:
  d=d.copy(); d.date=pd.to_datetime(d.date); raw[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in raw.items()}); r=cl.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date'])
col=[c for c in v.columns if c!='date'][0]; vs=v.set_index('date')[col].astype(float).reindex(cl.index).ffill()
vr=vs.pct_change(5).clip(lower=0,upper=1).fillna(0)
rawfac=-r.rolling(5,min_periods=3).sum().shift(1)
fac=rawfac.sub(rawfac.median(axis=1),axis=0).mul(1+vr,axis=0)
rows=[]
for dt in fac.index:
 vals=fac.loc[dt]
 if vals.notna().sum()<8: continue
 ix=cl.index.get_loc(dt)
 for h in [1,3,5,10]:
  if ix+h>=len(cl.index): continue
  z=pd.concat([vals.rename('f'),(cl.iloc[ix+h]/cl.iloc[ix]-1).rename('r')],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,len(z),z.f.corr(z.r,method='spearman')))
r=pd.DataFrame(rows,columns=['date','h','n','ic']); print('assets',len(raw),'dates',len(cl),'range',cl.index.min(),cl.index.max())
for h in [1,3,5,10]:
 x=r[r.h==h].dropna(); ic=x.ic
 print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f coverage %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean(),len(ic),x.n.mean(),x.n.mean()/len(U)))
 for label,cut in [('recent180',180),('recent360',360)]:
  y=ic.tail(cut); print(label,'%.6f %.6f'%(y.mean(),y.mean()/y.std(ddof=1)))
rank=fac.rank(axis=1,pct=True); rc=[]
for i in range(1,len(rank)):
 z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(z)>=8: rc.append(1-z.iloc[:,0].corr(z.iloc[:,1]))
print('turnover_proxy',np.nanmean(rc),'coverage',fac.notna().sum(axis=1).mean()/len(U))
out=[{'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(fac.loc[dt,s])} for dt in fac.index for s in fac.columns if pd.notna(fac.loc[dt,s])]
pd.DataFrame(out).to_csv('scripts/miner_2_20290531_vix_stress_residual_reversal_signal.csv',index=False)
print('artifact_rows',len(out))

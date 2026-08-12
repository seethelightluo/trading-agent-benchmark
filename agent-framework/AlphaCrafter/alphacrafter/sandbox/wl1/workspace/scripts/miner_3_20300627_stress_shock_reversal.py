import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof='2030-06-26'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in syms:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=asof]; px[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(px).sort_index(); r=c.pct_change()
v=get_index_daily_data('VIX',4000)
if v is None or len(v)==0: v=get_stock_daily_data('VIX',4000)
if v is not None and len(v):
 v=v.copy(); v.date=pd.to_datetime(v.date); v=v[v.date<=asof].set_index('date').close.astype(float); v=v.reindex(c.index).ffill()
else: v=pd.Series(index=c.index,dtype=float)
if v.notna().sum()<100: v=(-r.mean(axis=1)).rolling(3,min_periods=3).sum()
vr=v.pct_change(); vol=r.rolling(20,min_periods=15).std(); shock=-r.rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)+0.005)
vmed=v.rolling(60,min_periods=40).median(); stress=vr.rolling(3,min_periods=3).sum().clip(-1,1); level=((v/(vmed+1e-6))-1).clip(-1,1); f=(shock*(1+0.75*stress.clip(lower=0)+0.50*level.clip(lower=0))).shift(1)
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
 if h==1:
  for lab,a,b in [('2020-25','2020','2025-12-31'),('2026-28','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-06-26')]:
   x=q[(q.index>=a)&(q.index<=b)]; print(lab,len(x),f'IC={x.ic.mean():.6f}',f'ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f}')
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20300627_stress_shock_reversal_signal.csv',index=False); print('symbols',out.symbol.nunique(),'rows',len(out),'coverage',len(out)/(len(f)*len(syms)),'turnover',f.rank(pct=True).diff().abs().mean().mean())

import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=4500) for s in U}
px=pd.concat({s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None},axis=1).sort_index()
ms=[]
for name in ['VIX','DXY']:
 q=pd.read_csv('../persistent/index_data/'+name+'.csv'); q.date=pd.to_datetime(q.date)
 ms.append(q.drop_duplicates('date').set_index('date')['close'].astype(float).rename(name))
m=pd.concat(ms,axis=1).reindex(px.index).ffill().pct_change()
r=np.log(px).diff(); out=pd.DataFrame(index=px.index,columns=px.columns,dtype=float)
for i in range(60,len(px)):
 sl=slice(i-59,i+1); X=m.iloc[sl].to_numpy()
 if np.isfinite(X).all():
  X=np.column_stack([np.ones(len(X)),X])
  for s in px.columns:
   y=r[s].iloc[sl].to_numpy()
   if np.isfinite(y).all():
    b=np.linalg.lstsq(X,y,rcond=None)[0]
    out.loc[px.index[i],s]=r[s].iloc[sl].sum()-b[1:]@m.iloc[i].to_numpy()*60
vol=r.rolling(60).std().replace(0,np.nan)
f=(-out/vol).shift(1)
f.to_csv('scripts/miner_1_20330203_macro_residual_reversal_vixdxy_signal.csv',index_label='date')
for h in [10,20,40,60]:
 rows=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1: rows.append((px.index[i],len(z),z.x.corr(z.y,method='spearman')))
 a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); mu=a.ic.mean(); ir=mu/a.ic.std(ddof=1)*np.sqrt(len(a))
 recent=a[a.index>='2030-01-01']; rmu=recent.ic.mean(); rir=rmu/recent.ic.std(ddof=1)*np.sqrt(len(recent))
 print(f'h={h} dates={len(a)} avgN={a.n.mean():.2f} IC={mu:.6f} ICIR={ir:.6f} hit={(a.ic>0).mean():.4f} recent_dates={len(recent)} recentIC={rmu:.6f} recentICIR={rir:.6f}')
print('loaded',len(px.columns),'dates',len(px),'coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())

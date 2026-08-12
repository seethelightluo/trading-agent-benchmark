import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof='2030-03-06'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in syms:
 d=get_stock_daily_data(s,days=2800)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=2800)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d[d.date<=pd.Timestamp(asof)].sort_values('date');px[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(px).sort_index(); r=c.pct_change(); r20=c.pct_change(20); r60=c.pct_change(60); vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
# Relative trend: volatility-adjusted 20d return, centered by contemporaneous cross-sectional median;
# breadth gate keeps signal only when the median 20d return agrees with the asset direction.
raw=r20/(vol+0.01); med=r20.median(axis=1); f=raw.sub(raw.median(axis=1),axis=0); f=f.where(np.sign(r20).eq(np.sign(med),axis=0)).shift(1)
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
 for lab,a,b in [('2020-25','2020','2025-12-31'),('2026-28','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-03-06')]:
  x=q[(q.index>=a)&(q.index<=b)]
  if len(x):print(lab,len(x),f'{x.ic.mean():.6f}',f'{x.ic.mean()/x.ic.std(ddof=1):.6f}')
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20300307_relative_breadth_signal.csv',index=False);print('coverage',out.symbol.nunique(),'rows',len(out),'turnover',f.rank(pct=True).diff().abs().mean().mean())

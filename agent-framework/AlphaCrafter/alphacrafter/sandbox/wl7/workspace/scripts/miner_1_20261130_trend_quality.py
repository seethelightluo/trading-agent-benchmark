import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 try:d=get_index_daily_data(s,days=2600)
 except Exception:d=get_stock_daily_data(s,days=2600)
 x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);px[s]=x.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index();p=p.loc[p.index<=pd.Timestamp('2026-11-30')]
r=p.pct_change(); mom=r.rolling(20,min_periods=10).sum(); vol=r.rolling(20,min_periods=10).std(); cons=(r>0).astype(float).rolling(20,min_periods=10).mean()
breadth=mom.notna().sum(axis=1).replace(0,np.nan); breadth=(mom>0).sum(axis=1)/breadth
sig=((mom/vol.replace(0,np.nan))*(0.5+cons)).mul(0.75+0.5*(breadth-0.5),axis=0).shift(1)
print('shape',p.shape,'valid dates',sig.notna().any(axis=1).sum(),'coverage',sig.notna().mean().mean())
for h in [1,5,10,20]:
 fwd=p.shift(-h)/p-1; rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt].rename('a'),fwd.loc[dt].rename('b')],axis=1).dropna()
  if len(z)>=8:
   c=z.a.corr(z.b)
   if pd.notna(c):rows.append((dt,c,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']);m=q.ic.mean();sd=q.ic.std(ddof=1)
 print(f'h={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={m:.6f} ICIR={m/sd*np.sqrt(252):.6f} hit={(q.ic>0).mean():.4f}')
 if h==1:
  print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'range',q.date.min(),q.date.max())
  q['year']=q.date.dt.year;print(q.groupby('year').ic.agg(['mean','count']).round(5).tail(8).to_dict('index'))

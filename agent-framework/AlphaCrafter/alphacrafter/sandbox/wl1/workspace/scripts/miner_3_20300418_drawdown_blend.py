import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof='2030-04-17'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in syms:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d[d.date<=asof];px[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(px).sort_index(); r=c.pct_change(); v=r.rolling(40,min_periods=25).std()*np.sqrt(40)
trend=c.pct_change(20)/(v+0.01); short=-c.pct_change(5)/(v+0.01)
# Blend reversal for assets below their 60d high, trend for assets near highs; lag signal.
dd=c/c.rolling(60,min_periods=30).max()-1; gate=np.clip(-dd/0.20,0,1)
f=short*gate+trend*(1-gate); f=pd.DataFrame(f,index=c.index,columns=c.columns).shift(1)
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20300418_drawdown_blend_signal.csv',index=False);print('symbols',out.symbol.nunique(),'rows',len(out),'coverage',len(out)/(len(f)*len(syms)),'turnover',f.rank(pct=True).diff().abs().mean().mean())

import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof='2030-05-15'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,3200)
 if d is None or len(d)<150: d=get_index_daily_data(s,3200)
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d[d.date<=asof].set_index('date').close.astype(float)
px={s:load(s) for s in syms}; px={s:x for s,x in px.items() if x is not None and len(x)>0}
c=pd.DataFrame(px).sort_index(); r=c.pct_change();
# Breadth-conditioned short reversal: contrarian 5d move, volatility scaled,
# stronger when cross-asset 20d breadth is weak/strongly dispersed; lag 1 day.
vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
breadth=(r.rolling(20,min_periods=15).sum()>0).mean(axis=1)
disp=r.rolling(20,min_periods=15).sum().std(axis=1)
state=1+0.8*(0.5-breadth).abs()+0.5*disp/(r.rolling(60,min_periods=30).std().mean(axis=1)*np.sqrt(20)+0.01)
f=-(0.65*r.rolling(2).sum()+0.35*r.rolling(5).sum())/(vol+0.01)*state.values[:,None]
f=pd.DataFrame(f,index=c.index,columns=c.columns).rolling(2,min_periods=2).mean().shift(1)
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
 for lab,a,b in [('2020-25','2020','2025-12-31'),('2026-28','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-05-15')]:
  x=q[(q.index>=a)&(q.index<=b)]
  if len(x): print(lab,len(x),f'IC={x.ic.mean():.6f}',f'ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f}')
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20300516_breadth_reversal_signal.csv',index=False)
print('symbols',out.symbol.nunique(),'rows',len(out),'coverage',len(out)/(len(f)*len(syms)),'turnover',f.rank(pct=True).diff().abs().mean().mean())

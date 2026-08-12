import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof='2030-07-10'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,4000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d[d.date<=asof]; px[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(px).sort_index(); r=c.pct_change()
# Candidate: trend persistence. Risk-adjusted medium trend, rewarded only when 5/10/20d directions agree.
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
base=(c/c.shift(60)-1)/(vol+1e-8)
signs=pd.concat([(c/c.shift(k)-1).apply(np.sign) for k in [5,10,20]],axis=1,keys=['s5','s10','s20'])
# agreement score in [-1,1], retaining direction and penalizing disagreement
agree=signs.mean(axis=1,level=1) if False else None
s5=np.sign(c/c.shift(5)-1); s10=np.sign(c/c.shift(10)-1); s20=np.sign(c/c.shift(20)-1)
ag=(s5+s10+s20)/3
f=(base*ag).shift(1)
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
 for lab,a,b in [('2020-25','2020','2025-12-31'),('2026-28','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-07-10')]:
  x=q[(q.index>=a)&(q.index<=b)]
  if len(x): print(lab,len(x),f'IC={x.ic.mean():.6f}',f'ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f}')
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20300711_trend_persistence_signal.csv',index=False)
print('symbols',out.symbol.nunique(),'rows',len(out),'coverage',len(out)/(len(f)*len(syms)),'turnover',f.rank(pct=True).diff().abs().mean().mean())

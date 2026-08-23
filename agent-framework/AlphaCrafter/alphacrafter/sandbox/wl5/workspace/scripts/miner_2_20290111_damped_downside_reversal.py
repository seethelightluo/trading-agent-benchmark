import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); rows=[]
for t in p.index:
 if len(r.loc[:t])<25: continue
 rr=p.loc[t]/p.shift(5).loc[t]-1; trend=p.loc[t]/p.shift(20).loc[t]-1
 down=r.loc[:t].tail(20).clip(upper=0).std(); f=-rr/(down+1e-8)
 f=f*(1-0.5*np.sign(rr*trend).fillna(0)); f=f.replace([np.inf,-np.inf],np.nan)
 for s in U:
  if s in f and pd.notna(f[s]): rows.append((t,s,f[s]))
f=pd.DataFrame(rows,columns=['date','symbol','factor']).set_index(['date','symbol'])
print('rows',len(p),'instruments',len(px),'factor rows',len(f),'dates',f.index.get_level_values(0).nunique())
for h in [5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; dates=[]; cov=[]
 for t,g in f.groupby(level=0):
  if t not in fr.index: continue
  x=g.factor.droplevel(0); y=fr.loc[t].reindex(x.index); z=pd.concat([x,y.rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.factor.corr(z.y,method='spearman')); dates.append(t); cov.append(len(z)/len(U))
 a=np.array(vals); print(h,'obs',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),'hit',np.mean(a>0),'coverage',np.mean(cov))
 if h==10:
  for label,lo,hi in [('2020-24','2020','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-28','2027','2028-12-31'),('recent252',None,None)]:
   q=a[-252:] if label=='recent252' else np.array([v for d,v in zip(dates,a) if str(d)[:4]>=lo and str(d)<=hi])
   print('regime',label,'n',len(q),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/(np.nanstd(q,ddof=1)+1e-12),'hit',np.mean(q>0))
ranks=f.groupby(level=0).factor.rank(pct=True); print('turnover',ranks.unstack().diff().abs().mean(axis=1).mean())
f.reset_index().to_csv('scripts/miner_2_20290111_damped_downside_reversal_signal.csv',index=False)

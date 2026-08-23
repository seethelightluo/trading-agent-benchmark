import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); med=r.median(axis=1)
# Candidate: 3-day relative shock reversal, volatility normalized, gated to avoid
# buying assets in persistent 60d downtrends. All inputs lagged through t.
rel3=p.pct_change(3).sub(p.pct_change(3).median(axis=1),axis=0)
vol20=r.rolling(20).std(); trend60=p.pct_change(60)
f=(-rel3/(vol20*np.sqrt(3)+1e-8))
f=f.where(trend60>-0.15, f*0.25).replace([np.inf,-np.inf],np.nan)
rows=[]
for t in f.index:
 for s in U:
  if s in f.columns and pd.notna(f.loc[t,s]): rows.append((t,s,f.loc[t,s]))
f=pd.DataFrame(rows,columns=['date','symbol','factor']).set_index(['date','symbol'])
print('price_dates',len(p),'instruments',len(px),'factor_rows',len(f),'factor_dates',f.index.get_level_values(0).nunique())
for h in [5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; dates=[]; cov=[]
 for t,g in f.groupby(level=0):
  if t not in fr.index: continue
  x=g.factor.droplevel(0); y=fr.loc[t].reindex(x.index); z=pd.concat([x,y.rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(z.factor.corr(z.y,method='spearman')); dates.append(t); cov.append(len(z)/len(U))
 a=np.asarray(vals); print('horizon',h,'obs',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),'hit',np.mean(a>0),'coverage',np.mean(cov))
 if h==10:
  for label,lo,hi in [('2020-24','2020','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-28','2027','2028-12-31'),('recent252',None,None)]:
   q=a[-252:] if label=='recent252' else np.array([v for d,v in zip(dates,a) if str(d)[:4]>=lo and str(d)<=hi])
   print('regime',label,'n',len(q),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/(np.nanstd(q,ddof=1)+1e-12),'hit',np.mean(q>0))
ranks=f.groupby(level=0).factor.rank(pct=True)
print('turnover',ranks.unstack().diff().abs().mean(axis=1).mean())
f.reset_index().to_csv('scripts/miner_1_20290125_shock_reversal_signal.csv',index=False)

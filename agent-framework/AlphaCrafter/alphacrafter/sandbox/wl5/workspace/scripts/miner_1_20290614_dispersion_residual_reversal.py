import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
rel5=p.pct_change(5).sub(p.pct_change(5).median(axis=1),axis=0)
vol20=r.rolling(20).std()
csmean=r.mean(axis=1); disp=r.sub(csmean,axis=0).abs().mean(axis=1)
disp=disp.rolling(60).rank(pct=True)
f=(-rel5/(vol20*np.sqrt(5)+1e-8)).mul(0.5+disp,axis=0).replace([np.inf,-np.inf],np.nan)
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
 a=np.asarray(vals); print('horizon',h,'obs',len(a),'avg_n',np.mean(np.asarray(cov)*15),'IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(np.nanmean(a),np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),np.mean(a>0),np.mean(cov)))
 if h==10:
  for label,lo,hi in [('2020-24','2020','2024-12-31'),('2025-26','2025','2026-12-31'),('2027-28','2027','2028-12-31'),('recent252',None,None)]:
   q=a[-252:] if label=='recent252' else np.array([v for d,v in zip(dates,a) if str(d)[:4]>=lo and str(d)<=hi])
   print('regime',label,'n',len(q),'IC %.6f ICIR %.6f'%(np.nanmean(q),np.nanmean(q)/(np.nanstd(q,ddof=1)+1e-12)))
ranks=f.groupby(level=0).factor.rank(pct=True)
print('turnover',ranks.unstack().diff().abs().mean(axis=1).mean())
f.reset_index().to_csv('scripts/miner_1_20290614_dispersion_residual_reversal_signal.csv',index=False)

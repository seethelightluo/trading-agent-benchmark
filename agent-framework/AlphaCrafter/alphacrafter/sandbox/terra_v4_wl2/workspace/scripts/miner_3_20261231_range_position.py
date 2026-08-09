import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-31')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
# Range-location reversal: yesterday's close location within its high-low range, reversed.
rows=[]
for s,x in D.items():
    rng=(x.high-x.low).replace(0,np.nan); f=-(x.close-x.low)/rng; y=x.close.shift(-1)/x.close-1
    for dt in x.index:
        if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): rows.append((dt,s,f.loc[dt],y.loc[dt]))
a=pd.DataFrame(rows,columns=['date','symbol','factor','forward']); vals=[]; dates=[]; ns=[]
for dt,g in a.groupby('date'):
    if len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1: vals.append(spearmanr(g.factor,g.forward).statistic);dates.append(dt);ns.append(len(g))
z=np.array(vals); print('dates',len(z),'avg_names',np.mean(ns),'coverage',a.symbol.nunique()/15,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
r=a.assign(rank=a.groupby('date').factor.rank(pct=True)).pivot(index='date',columns='symbol',values='rank'); print('turnover',r.diff().abs().mean().mean())
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31')]:
 v=np.array([g.factor.corr(g.forward,method='spearman') for dt,g in a.groupby('date') if lo<=str(dt.date())<=hi and len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1]); print(label,len(v),v.mean(),v.mean()/v.std(ddof=1))
for h in [3,5,10]:
 v=[]
 for dt,g in a.groupby('date'):
  q=[]
  for s in U:
   x=D[s]; rng=(x.high-x.low).replace(0,np.nan); ff=-(x.close-x.low)/rng; yy=x.close.shift(-h)/x.close-1
   if dt in x.index and pd.notna(ff.loc[dt]) and pd.notna(yy.loc[dt]): q.append((ff.loc[dt],yy.loc[dt]))
  if len(q)>=8:v.append(spearmanr(np.array(q)[:,0],np.array(q)[:,1]).statistic)
 v=np.array(v);print('horizon',h,len(v),v.mean(),v.mean()/v.std(ddof=1))
out=a[['date','symbol','factor']].rename(columns={'factor':'signal'});out.to_csv('../persistent/factor_signals_miner_3_20261231_range_position.csv',index=False)

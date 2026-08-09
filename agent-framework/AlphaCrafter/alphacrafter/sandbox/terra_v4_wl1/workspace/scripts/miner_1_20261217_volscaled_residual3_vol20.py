import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]
 r=d.close.pct_change(); rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r3':d.close.pct_change(3),'v20':r.rolling(20,min_periods=10).std(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows); p=x.pivot(index='date',columns='symbol',values='r3'); med=p.median(axis=1); med[p.count(axis=1)<8]=np.nan
x['factor']=-(x.r3-x.date.map(med))/x.v20.replace(0,np.nan)
def metrics(q):
 vals=[]; ns=[]
 for _,g in q.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8: vals.append(spearmanr(g.factor,g.y).statistic); ns.append(len(g))
 a=np.array(vals); return {'dates':len(a),'avg_n':float(np.mean(ns)),'ic':float(np.mean(a)),'icir':float(np.mean(a)/np.std(a,ddof=1)),'hit':float(np.mean(a>0))}
print('universe',len(syms),'rows',len(x),'all',metrics(x))
for lo,hi,nm in [('2020','2022','2020-2022'),('2023','2024','2023-2024'),('2025','2026','2025-2026')]: print(nm,metrics(x[(x.date>=lo)&(x.date<=hi)]))
print('coverage',float(x.factor.notna().mean()),'turnover',float(x.dropna().pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
x[['date','symbol','factor']].dropna().to_csv('scripts/miner_1_20261217_volscaled_residual3_vol20_signal.csv',index=False)

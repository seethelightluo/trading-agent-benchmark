import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]
 r=d.close.pct_change(); rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r2':d.close.pct_change(2),'r4':d.close.pct_change(4),'r6':d.close.pct_change(6),'v':r.rolling(20,min_periods=15).std(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
for col in ['r2','r4','r6']:
 med=x.pivot(index='date',columns='symbol',values=col).median(axis=1); x['f'] = -(x[col]-x.date.map(med))/x.v
 print('\n',col)
 for h in [1,5]:
  z=x.copy(); z['yy']=z.groupby('symbol').close.transform('') if False else z.y
  # forward h
  if h>1:
   ys=[]
   for s in syms:
    d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; ys.append(pd.DataFrame({'date':d.date,'symbol':s,'yy':d.close.shift(-h)/d.close-1}))
   z=z[['date','symbol','f']].merge(pd.concat(ys),on=['date','symbol'])
  vals=[]; ns=[]
  for dt,g in z.groupby('date'):
   g=g.dropna(subset=['f','yy'])
   if len(g)>=8: vals.append(spearmanr(g.f,g.yy).statistic);ns.append(len(g))
  a=np.asarray(vals); print('H',h,'dates',len(a),'N',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 v=x.dropna(subset=['f']); print('coverage',len(v)/len(x))

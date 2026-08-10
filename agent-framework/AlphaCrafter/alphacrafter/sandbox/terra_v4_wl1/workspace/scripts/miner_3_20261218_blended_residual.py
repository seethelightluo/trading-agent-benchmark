import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy(); r=d.close.pct_change()
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r2':d.close.pct_change(2),'r3':d.close.pct_change(3),'r5':d.close.pct_change(5),'v':r.rolling(20,min_periods=10).std(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True); mats={k:x.pivot(index='date',columns='symbol',values=k) for k in ['r2','r3','r5','v']}; med={k:mats[k].median(axis=1) for k in ['r2','r3','r5']}
# blended, market-neutralized short-term reversal; all inputs lagged one completed session
f=0
for k,w in [('r2',.25),('r3',.5),('r5',.25)]: f += -w*(x[k]-x.date.map(med[k]))
x['factor']=f/x.v.replace(0,np.nan)
def calc(z):
 out=[]; ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['factor','y'])
  if len(g)>=8: out.append(spearmanr(g.factor,g.y).statistic);ns.append(len(g))
 a=np.asarray(out); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('UNIVERSE',len(U),'rows',len(x)); print('H1',calc(x))
for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026-12-17','25-26')]: print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
for h in [5,10]:
 yy=[]
 for s in U:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END]; yy.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 print('H',h,calc(x[['date','symbol','factor']].merge(pd.concat(yy),on=['date','symbol'])))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True); print('coverage',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean(),'artifact_rows',len(v));v[['date','symbol','factor']].to_csv('scripts/miner_3_20261218_blended_residual_signal.csv',index=False)

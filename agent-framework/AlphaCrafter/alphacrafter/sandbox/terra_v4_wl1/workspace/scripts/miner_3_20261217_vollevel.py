import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17');syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];a=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];r=d.close.pct_change();a.append(pd.DataFrame({'date':d.date,'symbol':s,'factor':-r.rolling(20,min_periods=15).std(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(a)
def c(z):
 q=[];n=[]
 for dt,g in z.groupby('date'):
  g=g.dropna();
  if len(g)>=8:q.append(spearmanr(g.factor,g.y).statistic);n.append(len(g))
 q=np.array(q);return len(q),np.mean(n),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
print('H1',c(x))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026-12-17')]:print(lo,hi,c(x[(x.date>=lo)&(x.date<=hi)]))
for h in [5,10]:
 z=[]
 for s in syms:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];z.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 print('H',h,c(x[['date','symbol','factor']].merge(pd.concat(z),on=['date','symbol'])))
v=x.dropna();r=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('coverage',len(v)/len(x),'turnover',r.diff().abs().mean(axis=1).mean());v.to_csv('scripts/miner_3_20261217_vollevel_signal.csv',index=False)

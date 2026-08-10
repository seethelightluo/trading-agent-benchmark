import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END]
 prev=d.close.shift(1); gap=d.open/prev-1; intr=d.close/d.open-1
 # overnight gap exhaustion: fade the opening gap, scaled by recent gap volatility
 gv=gap.rolling(20,min_periods=10).std(); f=-gap/gv.replace(0,np.nan)
 rows.append(pd.DataFrame({'date':d.date,'symbol':s,'f':f,'y':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
def calc(z,col='y'):
 a=[];ns=[]
 for dt,g in z.groupby('date'):
  g=g.dropna(subset=['f',col]);
  if len(g)>=8:a.append(spearmanr(g.f,g[col]).statistic);ns.append(len(g))
 a=np.array(a);return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),(a>0).mean()
print('universe',len(syms),'rows',len(x),'daily',calc(x),'5day',calc(x,'y5'))
for lo,hi,n in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026','25-26')]:print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
v=x.dropna(subset=['f']); print('coverage',len(v)/len(x));v[['date','symbol','f']].to_csv('scripts/miner_1_20261217_gap_exhaustion_signal.csv',index=False)

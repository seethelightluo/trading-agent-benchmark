import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-12-17'); rows=[]
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].set_index('date'); r=d.close.pct_change()
 # lagged downside-risk-adjusted medium momentum; all inputs end before decision
 down=r.where(r<0,0).rolling(20,min_periods=15).std().shift(1); mom=r.rolling(10,min_periods=8).sum().shift(1); f=mom/(down+1e-8)
 rows.append(pd.DataFrame({'date':d.index,'symbol':s,'factor':f,'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(rows,ignore_index=True); out=[]
for dt,g in x.groupby('date'):
 g=g.dropna();
 if len(g)>=8:
  ic=spearmanr(g.factor,g.y).statistic
  if np.isfinite(ic):out.append((dt,ic,len(g)))
a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); z=a.ic
print('UNIVERSE',len(U),'dates',len(z),'avg_n',a.n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026-12-17')]:
 q=z.loc[lo:hi]; print('REG',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [5,10]:
 ys=[]
 for s in U:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; ys.append(pd.DataFrame({'date':d.date,'symbol':s,'y':d.close.shift(-h)/d.close-1}))
 q=x[['date','symbol','factor']].merge(pd.concat(ys),on=['date','symbol']); out=[]
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8:
   v=spearmanr(g.factor,g.y).statistic
   if np.isfinite(v):out.append(v)
 v=np.array(out); print('H',h,'dates',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
print('coverage',x.factor.notna().mean()); x[['date','symbol','factor']].dropna().to_csv('scripts/miner_2_20261218_downside_mom_signal.csv',index=False)

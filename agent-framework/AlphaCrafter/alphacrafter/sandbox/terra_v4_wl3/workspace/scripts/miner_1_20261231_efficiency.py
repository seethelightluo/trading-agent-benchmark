import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# trend efficiency: signed 5d displacement divided by path length, designed to distinguish orderly trends from noisy moves
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 d=d[['date','close']].copy(); d['r']=d.close.pct_change()
 d['f']=(d.close/d.close.shift(5)-1)/(d.r.abs().rolling(5).sum()+1e-9)
 d['y']=d.close.shift(-1)/d.close-1
 d['symbol']=s; rows.append(d)
x=pd.concat(rows)
obs=[]; dates=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['f','y'])
 if len(g)>=8:
  obs.append(spearmanr(g.f,g.y).statistic); dates.append(dt)
ic=np.array(obs); print('dates',len(ic),'avg_names',np.mean([len(x[(x.date==d)&x.f.notna()&x.y.notna()]) for d in dates]))
print('IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(ic),np.nanmean(ic)/np.nanstd(ic,ddof=1),np.mean(ic>0)))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 z=ic[(np.array(dates)>=pd.Timestamp(a+'-01-01'))&(np.array(dates)<=pd.Timestamp(b+'-12-31'))]; print(a,b,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1))
# turnover: rank signal changes day to day, all available
wide=x.pivot(index='date',columns='symbol',values='f'); ranks=wide.rank(axis=1,pct=True); print('coverage',x.f.notna().mean(),'turnover',ranks.diff().abs().mean().mean())
# horizons decay
for h in [1,5,10]:
 z=[]
 for dt,g in x.groupby('date'):
  q=g.copy(); q['y']=q.close.shift(-h)/q.close-1 # wrong within symbol due g only one date
 # recompute
 vals=[]
 for s in U:
  d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d['r']=d.close.pct_change(); d['f']=(d.close/d.close.shift(5)-1)/(d.r.abs().rolling(5).sum()+1e-9); d['y']=d.close.shift(-h)/d.close-1; vals.append(d[['date','f','y','close']].assign(symbol=s))
 q=pd.concat(vals); cs=[]
 for dt,g in q.groupby('date'):
  g=g.dropna(subset=['f','y']);
  if len(g)>=8: cs.append(spearmanr(g.f,g.y).statistic)
 cs=np.array(cs); print('h',h,'n',len(cs),'IC',np.mean(cs),'ICIR',np.mean(cs)/np.std(cs,ddof=1))
# artifact
out=x[['date','symbol','f']].dropna(); out.to_csv('scripts/miner_1_20261231_efficiency_signal.csv',index=False)

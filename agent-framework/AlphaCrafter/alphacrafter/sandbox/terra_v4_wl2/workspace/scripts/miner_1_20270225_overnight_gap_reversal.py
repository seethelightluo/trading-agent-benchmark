import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'; d=pd.read_csv(p,parse_dates=['date']).sort_values('date')
 D[s]=d.set_index('date')
# gap reversal: factor known at close t, next-day return t+1; use open t / close t-1 and invert
rows=[]
for s,d in D.items():
 for t in d.index:
  if t not in d.index: continue
  ix=d.index.get_loc(t)
  if ix<1 or ix+1>=len(d): continue
  prev=d.iloc[ix-1]; cur=d.iloc[ix]; nxt=d.iloc[ix+1]
  gap=cur['open']/prev['close']-1
  f=-gap
  fr=nxt['close']/cur['close']-1
  rows.append((t,s,f,fr))
x=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
ics=x.groupby('date').apply(lambda z: z['factor'].corr(z['fwd'],method='spearman') if len(z)>=8 else np.nan).dropna()
print('dates',len(ics),'avg_n',x.groupby('date').size().mean(),'coverage',len(x)/sum(len(d) for d in D.values()))
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit', (ics>0).mean(),'std',ics.std())
for h in [1]: print('horizon',h)
# regimes split
for name,z in [('2020-22',ics[ics.index<'2023-01-01']),('2023-24',ics[(ics.index>='2023-01-01')&(ics.index<'2025-01-01')]),('2025+',ics[ics.index>='2025-01-01'])]: print(name,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>2 else np.nan)
# turnover rank changes
r=x.assign(rank=x.groupby('date')['factor'].rank(pct=True)).sort_values(['symbol','date'])
print('mean abs rank change',r.groupby('symbol')['rank'].diff().abs().mean())
# monthly ic
print('monthly mean',ics.resample('ME').mean().mean(),'months',ics.resample('ME').mean().notna().sum())

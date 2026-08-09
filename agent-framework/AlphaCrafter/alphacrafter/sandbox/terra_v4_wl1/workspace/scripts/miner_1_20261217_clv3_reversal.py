import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames=[]
for s in symbols:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date')
  rng=(d.high-d.low).replace(0,np.nan)
  # close location, smoothed over 3 completed sessions; contrarian signal
  d['sig']=-(d.close-d.low)/rng + 0.5
  d['sig']=d.sig.rolling(3,min_periods=2).mean()
  d['fwd']=d.close.shift(-1)/d.close-1
  frames.append(d[['date','sig','fwd']].assign(symbol=s))
x=pd.concat(frames)
rows=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8:
  rows.append((dt,len(g),spearmanr(g.sig,g.fwd).statistic))
r=pd.DataFrame(rows,columns=['date','n','ic']).sort_values('date')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
for name,z in [('all',r),('2020_22',r[r.date<'2023-01-01']),('2023_24',r[(r.date>='2023-01-01')&(r.date<'2025-01-01')]),('2025_26',r[r.date>='2025-01-01']),('recent120',r.tail(120))]:
 ic=z.ic.mean(); ir=ic/z.ic.std(ddof=1); print(name,'n',len(z),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((z.ic>0).mean(),4))
for h in [1,5,10]:
 # forward close h days; rebuild aligned by each symbol
 yy=[]
 for s in symbols:
  f='../persistent/stock_data/'+s+'.csv'
  if os.path.exists(f):
   d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); rng=(d.high-d.low).replace(0,np.nan); sig=(-(d.close-d.low)/rng+0.5).rolling(3,min_periods=2).mean(); fwd=d.close.shift(-h)/d.close-1
   yy.append(pd.DataFrame({'date':d.date,'sig':sig,'fwd':fwd}))
 q=pd.concat(yy); out=[]
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8: out.append(spearmanr(g.sig,g.fwd).statistic)
 out=np.array(out); print('h',h,'dates',len(out),'IC',out.mean(),'ICIR',out.mean()/out.std(ddof=1))
# rank turnover among common date signal ranks
wide=x.pivot(index='date',columns='symbol',values='sig'); ranks=wide.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).mean())
# artifact
x.pivot(index='date',columns='symbol',values='sig').to_csv('scripts/miner_1_20261217_clv3_reversal_signal.csv')

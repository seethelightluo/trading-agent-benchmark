import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv')
 d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
p=pd.concat(px,axis=1).sort_index().ffill()
r=p.pct_change()
# one interpretable idea: 20d momentum / 20d vol, active only when lagged breadth is decisively trending
mom=p.pct_change(20).shift(1); vol=r.rolling(20).std().shift(1)
breadth=(mom>0).sum(axis=1)/mom.notna().sum(axis=1)
active=(breadth>=.60)|(breadth<=.40)
f=mom/vol
f=f.where(active, np.nan)
# forward compounded returns, aligned from information date t
out=[]
for h in [1,3,5,10]:
 fr=p.shift(-h)/p-1
 vals=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(ic): vals.append((dt,ic,len(z),breadth.loc[dt]))
 q=pd.DataFrame(vals,columns=['date','ic','n','breadth']).set_index('date')
 if len(q):
  mean=q.ic.mean(); sd=q.ic.std(ddof=1); ir=mean/sd*np.sqrt(252) if sd else np.nan
  hit=(q.ic>0).mean(); print(f'h={h} dates={len(q)} avgN={q.n.mean():.2f} active={len(q)/len(f):.1%} IC={mean:.6f} ICIR={ir:.6f} hit={hit:.3f}')
  for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-06-24')]:
   z=q.loc[a:b]
   if len(z): print(' ',a,'n',len(z),'IC',round(z.ic.mean(),6),'IR',round(z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252),6))
# turnover proxy among active dates, rank signal
rank=f.rank(axis=1,pct=True)
turn=rank.diff().abs().mean(axis=1).where(active).mean()
print('coverage_active',f.notna().mean().mean(),'turnover_proxy',turn,'active_dates',active.sum(),'total_dates',len(f))

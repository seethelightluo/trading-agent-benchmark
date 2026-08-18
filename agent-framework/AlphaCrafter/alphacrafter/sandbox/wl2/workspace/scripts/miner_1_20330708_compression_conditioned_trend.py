import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
p=pd.concat(px,axis=1).sort_index().ffill(); r=p.pct_change()
# Lagged compression breakout: medium trend favored after quietness, with no lookahead.
mom=p.pct_change(20).shift(1); v10=r.rolling(10).std().shift(1); v60=r.rolling(60).std().shift(1)
f=(mom/(v60+1e-12))*(v10/(v60+1e-12)).pow(-1)
# Only use ordinary compression, avoiding extreme noisy observations.
active=(v10<v60).sum(axis=1)>=8
f=f.where(active)
print('candidate compression-conditioned risk trend')
for h in [1,3,5,10]:
 fr=p.shift(-h)/p-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(ic): rows.append((dt,ic,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 if len(q):
  sd=q.ic.std(ddof=1); print(f'h={h} dates={len(q)} avgN={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/sd*np.sqrt(252):.6f} hit={(q.ic>0).mean():.3f}')
  for a,b in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-07-08')]:
   z=q.loc[a:b]
   if len(z): print(' ',a,'n',len(z),'IC',z.ic.mean(),'IR',z.ic.mean()/z.ic.std(ddof=1)*np.sqrt(252) if len(z)>1 else np.nan)
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'active_dates',int(active.sum()),'total_dates',len(f),'turnover',rank.diff().abs().mean(axis=1).where(active).mean())
# artifact for deterministic audit
sig=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); sig.to_csv('scripts/miner_1_20330708_compression_conditioned_trend_signal.csv',index=False)

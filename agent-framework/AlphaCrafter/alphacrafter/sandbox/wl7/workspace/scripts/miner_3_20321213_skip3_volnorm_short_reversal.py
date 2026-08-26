import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x)>100:
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Candidate: delayed 5-session reversal, ending three sessions before signal,
# normalized by slower 60-session realized volatility for a more stable risk scale.
f=-(p.pct_change(5).shift(3))/r.rolling(60).std().shift(3)
rows=[]
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   vals.append(a.iloc[:,0].corr(a.iloc[:,1])); ns.append(len(a)); dates.append(dt)
 q=pd.Series(vals).dropna(); rows.append((h,len(q),np.mean(ns),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1),(q>0).mean(),q))
print('cutoff',p.index.max().date(),'calendar_dates',len(p),'assets',len(D),'avg_valid=%.2f'%f.notna().sum(axis=1).mean())
for h,n,avgn,ic,sd,ir,hit,q in rows: print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,n,avgn,ic,ir,hit))
rank=f.rank(axis=1,pct=True); ch=(rank.diff().abs().sum(axis=1)/rank.notna().sum(axis=1)).dropna()
print('coverage=%.4f turnover=%.4f'%(f.notna().mean().mean(),ch.mean()))
# regime thirds for admission horizon
q=rows[3][-1]; print('H20 thirds',*[round(x.mean(),6) for x in np.array_split(q,3)])
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_3_20321213_skip3_volnorm_short_reversal_signal.csv')
print('artifact=scripts/miner_3_20321213_skip3_volnorm_short_reversal_signal.csv')

import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x)>100:
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
base=-(p.pct_change(10).shift(10))/r.rolling(40).std().shift(10)
# Relative delayed reversal: emphasize assets whose lagged 20d return is below
# the contemporaneous cross-sectional median; all inputs are lagged.
rr=p.pct_change(20).shift(10)
med=rr.median(axis=1)
rel=rr.sub(med,axis=0)
gate=1.0+0.50*np.tanh(-rel/0.10)
f=base*gate
print('cutoff',p.index.max().date(),'calendar_dates',len(p),'assets',len(D),'avg_valid=%.2f'%f.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1])); ns.append(len(a))
 q=pd.Series(vals).dropna(); print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 if h==10: print('H10 thirds',*[round(x.mean(),6) for x in np.array_split(q,3)])
rank=f.rank(axis=1,pct=True); ch=(rank.diff().abs().sum(axis=1)/rank.notna().sum(axis=1)).dropna()
print('coverage=%.4f turnover=%.4f'%(f.notna().mean().mean(),ch.mean()))
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20330307_relative_gated_skip10_reversal_signal.csv')
print('artifact=scripts/miner_2_20330307_relative_gated_skip10_reversal_signal.csv')

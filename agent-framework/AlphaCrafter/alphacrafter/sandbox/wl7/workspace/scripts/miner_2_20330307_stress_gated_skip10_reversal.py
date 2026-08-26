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
# Stress-conditioned delayed reversal. Breadth is measured only through the
# same lagged endpoint; amplify reversal during broad weakness, retain 25%
# exposure otherwise to avoid discontinuities.
ret20=p.pct_change(20).shift(10)
breadth=(ret20>0).sum(axis=1)/ret20.notna().sum(axis=1)
gate=np.where(breadth<0.40,1.0,0.25)
f=base.mul(pd.Series(gate,index=p.index),axis=0)
print('cutoff',p.index.max().date(),'calendar_dates',len(p),'assets',len(D),'avg_valid=%.2f'%f.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1])); ns.append(len(a))
 q=pd.Series(vals).dropna(); print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 if h==10: print('H10 thirds',*[round(x.mean(),6) for x in np.array_split(q,3)])
rank=f.rank(axis=1,pct=True); ch=(rank.diff().abs().sum(axis=1)/rank.notna().sum(axis=1)).dropna()
print('coverage=%.4f turnover=%.4f stress_frac=%.4f'%(f.notna().mean().mean(),ch.mean(),(breadth<.4).mean()))
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20330307_stress_gated_skip10_reversal_signal.csv')
print('artifact=scripts/miner_2_20330307_stress_gated_skip10_reversal_signal.csv')

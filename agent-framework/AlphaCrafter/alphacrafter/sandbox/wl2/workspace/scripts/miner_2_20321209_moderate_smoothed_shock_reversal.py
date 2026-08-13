import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None and len(x)>100:return x
  except Exception: pass
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items()}).sort_index().groupby(level=0).last()
R=C.pct_change()
# Lagged, smoothed 3-day shock; EW downside risk; moderate dispersion conditioning.
r3=C.pct_change(3).shift(1)
down=R.clip(upper=0).pow(2).ewm(span=30,min_periods=20).mean().shift(1).pow(.5)
disp=R.rolling(5,min_periods=5).std().mean(axis=1).shift(1)
med=disp.rolling(90,min_periods=45).median()
ratio=(disp/(med+1e-12)).clip(.5,1.5)
# attenuate extremes: strongest in ordinary/moderate dispersion, bounded scalar
cond=(1-((ratio-1).abs()/0.5)).clip(0.25,1.0)
f=(-r3/(down+1e-8)).mul(cond,axis=0)
# validate one-day and multi-day forward returns, rank IC
rows=[]
for i in range(len(f)-10):
 q=pd.DataFrame({'f':f.iloc[i],'r':R.iloc[i+1]}).dropna()
 if len(q)>=8 and q.f.nunique()>1: rows.append((f.index[i],q.f.rank().corr(q.r.rank()),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=o.ic
print('assets',len(D),'dates',len(C),'IC_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
for a,b in [('2020','2025'),('2026','2029'),('2030','2032')]:
 q=ic.loc[a:b];print(a,b,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std()))
for h in [3,5,10]:
 rr=sum(R.shift(-k) for k in range(1,h+1)); vals=[]
 for i in range(len(f)-h):
  q=pd.DataFrame({'f':f.iloc[i],'r':rr.iloc[i]}).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append(q.f.rank().corr(q.r.rank()))
 print('decay',h,len(vals),np.nanmean(vals))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_2_20321209_moderate_smoothed_shock_reversal_signal.csv',index_label='date')

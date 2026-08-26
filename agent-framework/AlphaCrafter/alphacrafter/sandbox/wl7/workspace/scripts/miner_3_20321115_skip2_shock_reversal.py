import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x)>100:
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change(); vol40=r.rolling(40).std()
# Skip-2 volatility-normalized shock reversal: fade the 5-day move ending two sessions ago.
# Every feature is shifted two sessions, so the current and immediately prior close are excluded.
f=(-(p.pct_change(5).shift(2))/vol40.shift(2))
rows=[]
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1]))
 q=pd.Series(vals).dropna(); rows.append((h,len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1),(q>0).mean()))
print('cutoff',p.index.max().date(),'calendar_dates',len(p),'avgN=%.2f'%f.notna().sum(axis=1).mean())
for h,n,ic,sd,ir,hit in rows: print('H%d dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,n,ic,ir,hit))
rank=f.rank(axis=1,pct=True); ch=(rank.diff().abs().sum(axis=1)/rank.notna().sum(axis=1)).dropna()
print('coverage=%.4f turnover=%.4f'%(f.notna().mean().mean(),ch.mean()))
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_3_20321115_skip2_shock_reversal_signal.csv')

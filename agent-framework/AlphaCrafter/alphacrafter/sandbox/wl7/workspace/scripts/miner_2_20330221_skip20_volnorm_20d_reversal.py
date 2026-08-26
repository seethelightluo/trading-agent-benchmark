import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x)>150:
  z=x[['date','close']].copy(); z.date=pd.to_datetime(z.date); D[s]=z.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Medium horizon reversal, with a full 20-session information embargo and 80-session volatility scaling.
f=-(p.pct_change(20).shift(20))/r.rolling(80).std().shift(20)
print('cutoff',p.index.max().date(),'dates',len(p),'assets',len(D),'avg_valid=%.2f'%f.notna().sum(axis=1).mean())
for h in [1,5,10,20]:
 fr=p.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1])); ns.append(len(a))
 q=pd.Series(vals).dropna(); print('H%d dates=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 if h==10: print('H10 thirds',*[round(x.mean(),6) for x in np.array_split(q,3)])
rank=f.rank(axis=1,pct=True); ch=(rank.diff().abs().sum(axis=1)/rank.notna().sum(axis=1)).dropna()
print('coverage=%.4f turnover=%.4f'%(f.notna().mean().mean(),ch.mean()))
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20330221_skip20_volnorm_20d_reversal_signal.csv')
print('artifact=scripts/miner_2_20330221_skip20_volnorm_20d_reversal_signal.csv')

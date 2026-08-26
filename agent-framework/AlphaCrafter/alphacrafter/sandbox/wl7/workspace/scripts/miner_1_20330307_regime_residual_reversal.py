import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x)>100:
  z=x[['date','close']].copy();z.date=pd.to_datetime(z.date);D[s]=z.set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change();
vix=get_index_daily_data('VIX',days=5000)
v=vix.set_index(pd.to_datetime(vix.date)).close.astype(float).reindex(p.index).ffill()
# Delayed reversal, cross-sectionally residualized, with smooth volatility-regime weight.
raw=p.pct_change(10).shift(10); vol=r.rolling(40).std().shift(10)
base=-raw/vol
cs=base.mean(axis=1); f=base.sub(cs,axis=0)
z=(v-v.rolling(252).mean())/v.rolling(252).std()
weight=(1+0.35*z.clip(-2,2)).clip(.35,1.9)
f=f.mul(weight,axis=0)
print('cutoff',p.index.max().date(),'dates',len(p),'assets',len(D),'avg_valid',round(f.notna().sum(axis=1).mean(),2))
for h in [5,10,20]:
 fr=p.pct_change(h).shift(-h); q=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:q.append(a.iloc[:,0].corr(a.iloc[:,1]));ns.append(len(a))
 q=pd.Series(q);print('H%d obs=%d avgN=%.2f IC=%+.6f ICIR=%+.6f hit=%.4f'%(h,len(q),np.mean(ns),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 print(' thirds',*[round(x.mean(),6) for x in np.array_split(q,3)])
rank=f.rank(axis=1,pct=True);turn=(rank.diff().abs().sum(axis=1)/rank.notna().sum(axis=1)).dropna()
print('coverage=%.4f turnover=%.4f'%(f.notna().mean().mean(),turn.mean()))
f.index.name='date'; f.to_csv('scripts/miner_1_20330307_regime_residual_reversal_signal.csv')
print('artifact=scripts/miner_1_20330307_regime_residual_reversal_signal.csv')

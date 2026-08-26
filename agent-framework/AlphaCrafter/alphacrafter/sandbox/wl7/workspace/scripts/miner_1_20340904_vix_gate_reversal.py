import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:get_stock_daily_data(a,days=5000) for a in assets}
v=get_index_daily_data('VIX',days=5000)
P=pd.DataFrame({a:d.set_index('date')['close'] for a,d in px.items() if d is not None}).sort_index().ffill()
V=v.set_index('date')['close'].reindex(P.index).ffill()
r5=P.pct_change(5); f=-(r5).shift(1)
vg=(V.shift(1)/V.shift(61)-1).clip(-1,1)
gate=1+0.75*(vg>0).astype(float); f=f.mul(gate,axis=0)
print('cutoff',P.index.max().date(),'dates',len(P),'assets',P.shape[1])
for h in [1,5,10,20]:
 fr=P.pct_change(h).shift(-h); ics=[]; ns=[]
 for dt in f.index:
  x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: ics.append(x[ok].corr(y[ok],method='spearman')); ns.append(ok.sum())
 s=pd.Series(ics).dropna(); print('H%d dates=%d avgN=%.1f IC=%.6f ICIR=%.6f hit=%.3f'%(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
fr=P.pct_change(10).shift(-10); ics=[]
for dt in f.index:
 x=f.loc[dt];y=fr.loc[dt];ok=x.notna()&y.notna()
 if ok.sum()>=8: ics.append(x[ok].corr(y[ok],method='spearman'))
s=pd.Series(ics).dropna()
for w in [180,500,750]:
 q=s.tail(w);print('recent',w,'IC %.6f ICIR %.6f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(pct=True).diff().abs().mean().mean())
f.index.name='date';f.to_csv('scripts/miner_1_20340904_vix_gate_reversal_signal.csv')

import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:get_stock_daily_data(a,5000) for a in A};P=pd.DataFrame({a:d.set_index('date')['close'] for a,d in D.items()}).sort_index().ffill(); R=P.pct_change();
# volatility-normalized 5d reversal, lagged; interpretable risk-adjusted short-term mean reversion
f=(-(P.pct_change(5))/R.rolling(20).std()).shift(1)
print('cutoff',P.index.max().date(),'dates',len(P),'assets',P.shape[1])
for h in [1,5,10,20]:
 y=P.pct_change(h).shift(-h); z=[]; ns=[]
 for dt in f.index:
  x=f.loc[dt];q=y.loc[dt];ok=x.notna()&q.notna()
  if ok.sum()>=8:z.append(x[ok].corr(q[ok],method='spearman'));ns.append(ok.sum())
 s=pd.Series(z).dropna();print('H%d dates=%d avgN=%.1f IC=%.6f ICIR=%.6f hit=%.3f'%(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
y=P.pct_change(10).shift(-10);z=[]
for dt in f.index:
 x=f.loc[dt];q=y.loc[dt];ok=x.notna()&q.notna()
 if ok.sum()>=8:z.append(x[ok].corr(q[ok],method='spearman'))
s=pd.Series(z).dropna()
for w in [180,500,750]:
 q=s.tail(w);print('recent',w,'IC %.6f ICIR %.6f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(pct=True).diff().abs().mean().mean())
f.index.name='date';f.to_csv('scripts/miner_1_20340904_volnorm_reversal_signal.csv')

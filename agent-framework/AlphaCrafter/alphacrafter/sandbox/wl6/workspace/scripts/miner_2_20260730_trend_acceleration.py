import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'
def p(a):
 x=pd.read_csv(f'{B}/{a}.csv'); x.date=pd.to_datetime(x.date); return x.set_index('date').close.astype(float)
px=pd.concat({a:p(a) for a in A},axis=1).sort_index(); px=px.loc[:'2026-07-30']; r=px.pct_change()
# acceleration: recent 20d trend minus preceding 20d trend, normalized by trailing 60d vol
f=(px.pct_change(20)-px.pct_change(40).shift(20))*1.0/r.rolling(60).std()
print('factor trend_accel_20_40')
for h in [1,5,10]:
 y=px.shift(-h)/px-1; ic=[]; ns=[]; tr=[]; prev=None
 for d in f.index:
  z=f.loc[d]; q=y.loc[d]; ok=z.notna()&q.notna()
  if ok.sum()>=8:
   ic.append(spearmanr(z[ok],q[ok]).statistic);ns.append(ok.sum())
   if prev is not None:
    pp=prev.notna()&ok; tr.append((z[pp].rank(pct=True)-prev[pp].rank(pct=True)).abs().mean())
   prev=z
 a=np.array(ic); print(h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(np.mean(a),5),'ICIR',round(np.mean(a)/np.std(a,ddof=1),5),'hit',round(np.mean(a>0),4),'turn',round(np.mean(tr),4))
print('coverage',round(f.notna().sum(axis=1).mean()/15,4))
for name,v in {'mom20':px.pct_change(20),'rev5':-px.pct_change(5),'clv':(px-px.rolling(2).min())/(px.rolling(2).max()-px.rolling(2).min())}.items():
 z=pd.concat([f.stack().rename('f'),v.stack().rename('x')],axis=1).dropna();print('corr',name,round(z.corr(method='spearman').iloc[0,1],4))

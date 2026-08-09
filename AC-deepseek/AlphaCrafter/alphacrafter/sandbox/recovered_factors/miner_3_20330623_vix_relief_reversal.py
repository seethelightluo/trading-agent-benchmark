import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close']
    D[a]=x
p=pd.DataFrame(D).sort_index(); r=p.pct_change();
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# Candidate: short-term reversal damped in stressed/rising volatility; 5d reversal / 20d vol, with VIX relief confirmation
# positive signal means expected next return; use only info at t, evaluate t+1 onward
rv=r.rolling(20).std(); rev=-(p.pct_change(5))/rv
vixrel=-(vix.pct_change(5)).clip(-.5,.5) # positive on VIX relief
# bounded multiplier: reversal is stronger when VIX is falling, but avoid sign flip
mult=(1+0.35*np.tanh(vixrel*8))
s=rev.mul(mult,axis=0).shift(1)
rets={h:p.pct_change(h).shift(-h) for h in [1,5,10,20]}
print('period',p.index.min().date(),p.index.max().date(),'assets',len(assets))
print('valid cells',int(s.notna().sum().sum()),'coverage',s.notna().mean().mean())
for h,y in rets.items():
  vals=[]; ns=[]
  for dt in s.index:
    z=pd.concat([s.loc[dt],y.loc[dt]],axis=1).dropna()
    if len(z)>=8:
      vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
  a=np.array(vals); print('H',h,'dates',len(a),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
# subperiods H1
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2033')]:
 vals=[]
 for dt in s.loc[lo:hi].index:
  z=pd.concat([s.loc[dt],rets[1].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(vals); print(lo,hi,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1) if len(a)>1 else np.nan)
# turnover rank proxy
ranks=s.rank(axis=1,pct=True); print('turnover',ranks.diff(10).abs().mean().mean())

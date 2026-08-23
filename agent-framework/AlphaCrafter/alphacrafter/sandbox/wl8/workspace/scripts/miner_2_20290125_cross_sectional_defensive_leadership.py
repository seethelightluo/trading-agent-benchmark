import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 d=get_stock_daily_data(s,2600); d.date=pd.to_datetime(d.date); fs[s]=d.set_index('date').close.astype(float)
p=pd.concat(fs,axis=1).sort_index(); p.columns=U; r=p.pct_change(); eq=r.mean(axis=1)
# Defensive leadership: relative 20d return of safe assets (gold and yields) versus broad cross-asset return, scaled by own vol.
safe=p[['XAU','US10Y','CN10Y']].pct_change(20).mean(axis=1); broad=(1+eq).rolling(20).apply(np.prod,raw=True)-1
sig=((safe-broad)/eq.rolling(20).std()).shift(1).clip(-8,8)
# broadcast regime signal with asset-specific defensive preference: safe assets get positive score, risk assets negative
base=p.pct_change(20).shift(1).div(r.rolling(20).std().shift(1)).clip(-8,8)
for s in ['XAU','US10Y','CN10Y']: base[s]=base[s]+sig
for s in U:
 if s not in ['XAU','US10Y','CN10Y']: base[s]=base[s]-sig
sig=base
for h in [1,3,5,10]:
 y=p.shift(-h)/p-1;a=[];ns=[]
 for dt in sig.index:
  q=sig.loc[dt];z=y.loc[dt];ok=q.notna()&z.notna()
  if ok.sum()>=8:a.append(q[ok].corr(z[ok]));ns.append(ok.sum())
 a=np.asarray(a); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4))
print('coverage',round(sig.notna().sum().sum()/(len(sig)*15),4))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20290125_defensive_leadership_signal.csv',index=False)

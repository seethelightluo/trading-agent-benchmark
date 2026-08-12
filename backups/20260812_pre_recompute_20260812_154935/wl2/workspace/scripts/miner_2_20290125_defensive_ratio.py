import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; F={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=4000)
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);F[s]=d.drop_duplicates('date').set_index('date').sort_index()
px=pd.DataFrame({s:d.close for s,d in F.items()});r=np.log(px).diff()
# Stable defensive carry proxy: medium horizon return penalized by downside-volatility
# and smoothed by 20/60 volatility ratio. Lagged one bar.
dn=np.sqrt((r.clip(upper=0)**2).rolling(60).mean()); tot=r.rolling(60).std()
sig=(r.rolling(20).sum()/(dn+1e-8))*(tot/(tot.rolling(60).mean()+1e-8))
f=sig.shift(1); f.to_csv('scripts/miner_2_20290125_defensive_ratio_signal.csv')
for h in [1,5,10,20]:
 y=r.shift(-h).rolling(h).sum() if h>1 else r.shift(-1); a=[];ds=[];ns=[];prev=None;tu=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8:
   q=spearmanr(f.loc[dt][ok],y.loc[dt][ok]).statistic
   if np.isfinite(q):
    a.append(q);ds.append(dt);ns.append(ok.sum()); z=f.loc[dt][ok].rank(pct=True);tu.append(np.mean(np.abs(z-(prev if prev is not None else z))));prev=z
 a=np.array(a);print('h',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turn',np.mean(tu))
 for st in ['2020-01-01','2023-01-01','2026-01-01','2027-01-01']:
  b=a[np.array(ds)>=pd.Timestamp(st)];print(st,len(b),b.mean(),b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
print('assets',len(F),'range',px.index.min(),px.index.max(),'coverage',px.notna().mean().mean())

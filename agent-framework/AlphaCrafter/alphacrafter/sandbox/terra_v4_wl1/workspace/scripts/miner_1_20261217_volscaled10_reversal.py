import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); px[s]=z.drop_duplicates('date').set_index('date').close
close=pd.DataFrame(px).sort_index(); r=close.pct_change()
# Volatility-scaled medium-horizon reversal, using only completed bars.
f=(-close.pct_change(10)/(np.sqrt(10)*r.rolling(30,min_periods=10).std())).replace([np.inf,-np.inf],np.nan)
fw1=close.pct_change(1).shift(-1)
for h in [1,5,10]:
 fw=close.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1: vals.append(a.f.corr(a.r));ns.append(len(a));ds.append(dt)
 ic=pd.Series(vals,index=ds).dropna(); print('h',h,'dates',len(ic),'avgN',round(np.mean(ns),2),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1)*np.sqrt(252),'hit',(ic>0).mean())
# diagnostics
print('coverage',f.notna().sum().sum()/f.size,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'period',close.index.min(),close.index.max(),'names',close.shape[1])
for label,mask in [('2020-22',f.index<'2023-01-01'),('2023-24',(f.index>='2023-01-01')&(f.index<'2025-01-01')),('2025-26',f.index>='2025-01-01')]:
 z=[]
 for dt in f.index[mask]:
  a=pd.DataFrame({'f':f.loc[dt],'r':fw1.loc[dt]}).dropna()
  if len(a)>=8:z.append(a.f.corr(a.r))
 z=pd.Series(z).dropna();print(label,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1)*np.sqrt(252) if len(z)>1 else np.nan)
# signal artifact for deterministic audit
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20261217_volscaled10_reversal_signal.csv',index=False)

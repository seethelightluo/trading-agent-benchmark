import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=3000)
 if d is not None: D[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Novel candidate: recovery-confirmation score. Assets that are rebounding from a 60d drawdown,
# but only when the rebound is confirmed by positive 5d-vs-20d trend; scale by idiosyncratic risk.
vol=r.rolling(20,min_periods=15).std(); dd=p/p.rolling(60,min_periods=40).max()-1
rebound=r.rolling(5,min_periods=4).sum()
confirm=r.rolling(5,min_periods=4).sum()-r.rolling(20,min_periods=15).sum()/4
f=(rebound+0.35*confirm)*(-dd).clip(lower=0.0)/vol
rows=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=a.ic
print('candidate recovery_confirmation')
print('dates',len(q),'avgN',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for nm,mask in [('2020-22',a.index<'2023-01-01'),('2023-25',(a.index>='2023-01-01')&(a.index<'2026-01-01')),('2026-30',a.index>='2026-01-01')]:
 z=a.loc[mask].ic; print(nm,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
for h in [3,5,10]:
 y=p.pct_change(h).shift(-h+1); vals=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],y.iloc[i]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'IC',np.nanmean(vals),'dates',len(vals))
f.to_csv('scripts/miner_1_20300822_recovery_confirmation_signal.csv')

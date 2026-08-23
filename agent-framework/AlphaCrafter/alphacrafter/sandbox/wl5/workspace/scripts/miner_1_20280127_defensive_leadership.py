import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2028-01-26')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date')
d={s:load(s) for s in U}; c=pd.concat({s:x.close for s,x in d.items()},axis=1).sort_index(); r=c.pct_change()
# Defensive leadership trend: 20d relative strength, activated only when defensive basket leads global equities.
ret=c/c.shift(20)-1; eq=ret[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1); de=ret[['XAU','US10Y','CN10Y']].mean(axis=1); gate=(de>eq).astype(float)
f=ret.sub(ret.median(axis=1),axis=0).mul(gate,axis=0)
for h in [1,5,10,20]:
 y=c.pct_change(h).shift(-h); a=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 a=np.array(a); print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027),(2028,2028)]:
  q=a[[lo<=x.year<=hi for x in ds]]; print('reg',lo,hi,len(q),round(q.mean(),6) if len(q) else None)
print('coverage',round(f.notna().mean().mean(),4),'active_dates',int(gate.sum()),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20280127_defensive_leadership_signal.csv',index=False)

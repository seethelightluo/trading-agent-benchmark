import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2027-11-17')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date')
d={s:load(s) for s in U}; close=pd.concat({s:x.close for s,x in d.items()},axis=1).sort_index(); high=pd.concat({s:x.high for s,x in d.items()},axis=1).sort_index(); low=pd.concat({s:x.low for s,x in d.items()},axis=1).sort_index()
r=close.pct_change(); rng=((high-low)/close).rolling(20,min_periods=15).mean();
# Fade a persistent five-session displacement, normalized by typical daily range.
f=-(r.rolling(5,min_periods=5).sum()/(rng*np.sqrt(5)+1e-12)); f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10,20]:
 y=close.pct_change(h).shift(-h); a=[];ns=[];ds=[]
 for date in f.index:
  z=pd.DataFrame({'f':f.loc[date],'y':y.loc[date]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(date)
 a=np.array(a); print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
  q=a[[lo<=x.year<=hi for x in ds]];print('reg',lo,hi,len(q),round(q.mean(),6) if len(q) else None)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_3_20271118_range_shock_5d_signal.csv',index=False)

import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
E=pd.Timestamp('2027-11-17')
def load(s):
 return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').close
p=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=p.pct_change()
# Recovery score: position above 60-session low, penalized by current 10-session pullback.
low=p.rolling(60,min_periods=40).min(); high=p.rolling(60,min_periods=40).max()
recovery=(p/low-1) - 0.5*(p.pct_change(10))
# Cross-sectional IC of higher recovery score vs future returns
for h in [1,5,10,20]:
 y=p.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for d in recovery.index:
  z=pd.DataFrame({'f':recovery.loc[d],'y':y.loc[d]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); dates.append(d)
 a=np.asarray(vals); print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
  q=a[[lo<=d.year<=hi for d in dates]]; print('reg',lo,hi,len(q),round(q.mean(),6) if len(q) else None)
print('coverage',round(recovery.notna().mean().mean(),4),'turnover',round(recovery.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
recovery.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20271118_drawdown_recovery_signal.csv',index=False)

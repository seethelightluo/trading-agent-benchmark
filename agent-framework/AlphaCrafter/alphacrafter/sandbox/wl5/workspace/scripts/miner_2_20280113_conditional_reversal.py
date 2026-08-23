import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
E=pd.Timestamp('2028-01-12')
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 return pd.read_csv(p,parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').close
close=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=close.pct_change()
# Conditional medium-term reversal: reverse 20-session return only when cross-asset 20d median return is negative.
breadth=r.rolling(20,min_periods=15).median().sum(axis=1)
gate=(breadth<0)
f=-r.rolling(20,min_periods=20).sum().where(gate, np.nan)
f=f.sub(f.median(axis=1),axis=0)
y=close.pct_change(10).shift(-10)
ics=[]; ns=[]; ds=[]
for d in f.index:
 z=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
  ics.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(d)
a=np.array(ics); print('end',E.date(),'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2027),(2027,2028)]:
 q=a[[lo<=x.year<=hi for x in ds]]
 print('regime',lo,hi,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('coverage',round(f.notna().mean().mean(),4),'gate_days',int(gate.sum()),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_2_20280113_conditional_reversal_signal.csv',index=False)

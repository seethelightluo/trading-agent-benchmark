import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,days=5000)
            if x is not None and len(x): return x
        except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change()
# Observation-only DXY is lagged one day; signal is volatility-scaled 5d reversal in a dollar-strength/equity-stress regime.
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(px.index).ffill()
eq= r[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']]
breadth=eq.lt(0).sum(axis=1)/eq.notna().sum(axis=1)
dollar=dxy.pct_change(5)
vol=r.rolling(20).std()*np.sqrt(252)
f=(-r.rolling(5).sum()/vol).where((breadth.shift(1)>=.50)&(dollar.shift(1)>0))
fr={h:px.shift(-h)/px-1 for h in [1,5,10]}
obs=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr[1].loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: obs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
a=np.asarray(obs)
for h in [1,5,10]:
 vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=np.asarray(vals); print('H',h,'dates',len(q),'avgN',np.mean(ns) if h==1 else 'NA','IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1),'hit',np.mean(q>0))
for name,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-12-31')]:
 q=[x for dt,x in zip(f.index,obs) if str(dt)[:10]>=lo and str(dt)[:10]<=hi];q=np.asarray(q);print(name,'dates',len(q),'IC',np.mean(q) if len(q) else np.nan,'ICIR',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
print('active dates',int(f.notna().any(axis=1).sum()),'total',len(f),'coverage',f.notna().sum().sum()/(len(U)*len(f)),'avg active N',np.mean(ns))
ranks=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(ranks)):
 if f.iloc[i].notna().sum()>=8 and f.iloc[i-1].notna().sum()>=8: turn.append(np.mean(abs(ranks.iloc[i]-ranks.iloc[i-1]).dropna()))
print('turnover',np.mean(turn) if turn else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_dollar_stress_reversal.csv',index=False)

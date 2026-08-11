import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
asof='2028-05-18'
syms=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in syms:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<80: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); frames[s]=d.set_index('date').sort_index()
p=pd.DataFrame({s:d.close for s,d in frames.items()})
# Risk-adjusted trend with drawdown resilience: medium-term return divided by realized risk,
# then penalized by distance from the 60-day high. Signal is lagged one session.
r20=p.pct_change(20); v20=p.pct_change().rolling(20,min_periods=15).std(); dd=p/p.rolling(60,min_periods=40).max()-1
factor=((r20/(v20+1e-8))*(1+dd.clip(-1,0))).shift(1)
rows=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],p.pct_change(20).shift(-20).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
r=pd.DataFrame(rows,columns=['date','n','ic']).dropna()
print('symbols',len(frames),'dates',len(r),'avgN',round(r.n.mean(),2),'coverage',round(len(frames)/15,4))
for h in [5,10,20]:
 a=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],p.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a).dropna();print('horizon',h,'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
print('turnover',factor.rank(axis=1,pct=True).diff().abs().mean().mean())
for label,sub in [('2026+',r[r.date>='2026-01-01']),('2027+',r[r.date>='2027-01-01']),('2028YTD',r[r.date>='2028-01-01'])]:
 print(label,'dates',len(sub),'IC %.6f ICIR %.6f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1)))
r.to_csv('scripts/miner_1_20280518_risk_adjusted_drawdown_trend_signal.csv',index=False)

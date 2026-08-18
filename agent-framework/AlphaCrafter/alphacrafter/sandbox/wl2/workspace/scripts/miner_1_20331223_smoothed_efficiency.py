import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=4100)
 if d is None or len(d)<100: d=get_index_daily_data(s,days=4100)
 if d is not None: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=np.log(P).diff()
# Smoothed multi-scale path efficiency: average of lagged 10d and 40d directional efficiency.
def eff(n): return (np.log(P/P.shift(n))/(r.abs().rolling(n,min_periods=max(8,n-2)).sum())).shift(1)
sig0=(eff(10)+eff(40))/2
sig=sig0.rank(axis=1,pct=True).sub(.5).rolling(3,min_periods=2).mean()
vals=[]
for dt in sig.index:
 z=sig.loc[dt]; y=np.log(P.shift(-10)/P).loc[dt]; ok=z.notna()&y.notna()
 if ok.sum()>=8: vals.append((dt,z[ok].corr(y[ok]),int(ok.sum())))
q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
print('assets',len(px),'dates',len(P),'valid_dates',len(q),'avg_n',q.n.mean())
print('IC %.9f ICIR %.9f hit %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252),(q.ic>0).mean()))
for label,sub in [('early',q.loc[:'2025-12-31']),('mid',q.loc['2026-01-01':'2029-12-31']),('recent',q.loc['2030-01-01':])]:
 print(label,len(sub),'IC %.9f ICIR %.9f hit %.6f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1)*np.sqrt(252),(sub.ic>0).mean()))
print('coverage',sig.notna().sum().sum()/(len(sig)*len(U)),'turnover',sig.diff().abs().mean().mean())
sig.to_csv('scripts/miner_1_20331223_smoothed_efficiency_signal.csv',index_label='date')
q.to_csv('scripts/miner_1_20331223_smoothed_efficiency_10d_ic.csv')

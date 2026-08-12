import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<150:d=get_index_daily_data(s,2800)
 if d is not None:px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill();r=P.pct_change(); vol=r.rolling(20).std()*np.sqrt(252)
# Short-horizon residual reversal, scaled by volatility and attenuated when medium trend is strong.
raw=-P.pct_change(5)/(vol+1e-8); trend=P.pct_change(60)/(r.rolling(60).std()*np.sqrt(252)+1e-8)
f=(raw/(1+0.5*abs(trend))).shift(1)
for h in [1,5,10,20]:
 fr=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');m=q.ic.mean();sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'avgN %.2f'%q.n.mean(),'IC %.6f'%m,'ICIR %.6f'%(m/sd),'hit %.4f'%((q.ic>0).mean()))
 for label,sub in [('2020-2025',q.loc[:'2025-12-31']),('2026+',q.loc['2026-01-01':]),('2029+',q.loc['2029-01-01':]),('2030YTD',q.loc['2030-01-01':])]:
  if len(sub)>20:print(' ',label,'n',len(sub),'IC %.6f'%sub.ic.mean(),'ICIR %.6f'%(sub.ic.mean()/sub.ic.std(ddof=1)))
print('coverage %.4f'%(f.notna().sum(axis=1).mean()/len(U)),'turnover %.6f'%((f.rank(axis=1,pct=True).diff().abs().sum(axis=1)/2).mean()))
print('period',P.index.min(),P.index.max(),'assets',len(U));f.to_csv('scripts/miner_2_20300822_attenuated_reversal_signal.csv')

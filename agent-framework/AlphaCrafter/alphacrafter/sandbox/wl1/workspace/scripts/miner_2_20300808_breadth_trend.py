import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<150:d=get_index_daily_data(s,2600)
 if d is not None:px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Breadth-confirmed medium trend: 40d trend, conditioned on 20d breadth, volatility scaled and lagged.
breadth=(r.rolling(20).mean()>0).mean(axis=1)
trend=P.pct_change(40); vol=r.rolling(30).std()*np.sqrt(252)
# continuation in broad risk-on, defensive reversal in broad risk-off (interpretable)
sign=np.where(breadth.values[:,None]>=.55,1.,-0.35)
f=trend/(vol+1e-8)*sign
f=pd.DataFrame(f,index=P.index,columns=P.columns).shift(1)
for h in [1,5,10,20]:
 fr=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');m=q.ic.mean();sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'avgN',q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(m,m/sd,(q.ic>0).mean()))
print('coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',(f.rank(axis=1,pct=True).diff().abs().sum(axis=1)/2).mean())
for label,sub in [('2020-25',q.loc[:'2025']),('2026+',q.loc['2026':]),('2028+',q.loc['2028':]),('2029+',q.loc['2029':]),('2030',q.loc['2030':])]:
 if len(sub):print(label,'dates',len(sub),'IC',sub.ic.mean(),'ICIR',sub.ic.mean()/sub.ic.std(ddof=1))
f.to_csv('scripts/miner_2_20300808_breadth_trend_signal.csv')

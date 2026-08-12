import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None or len(d)<150: d=get_index_daily_data(s,2800)
 if d is not None: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Multi-horizon trend agreement: volatility-normalized returns at 20/60/120d,
# with a continuous agreement multiplier (avoids hard regime gates).
vol=r.rolling(40).std()*np.sqrt(252)
z20=P.pct_change(20)/(vol+1e-8); z60=P.pct_change(60)/(vol+1e-8); z120=P.pct_change(120)/(vol+1e-8)
agree=(np.sign(z20)+np.sign(z60)+np.sign(z120))/3
f=(0.25*z20+0.45*z60+0.30*z120)*(0.55+0.45*agree)
f=f.shift(1)
frs={}
for h in [1,5,10,20]:
 fr=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); frs[h]=q
 m=q.ic.mean(); sd=q.ic.std(ddof=1)
 print('horizon',h,'dates',len(q),'avgN %.2f'%q.n.mean(),'IC %.6f'%m,'ICIR %.6f'%(m/sd if sd else np.nan),'hit %.4f'%((q.ic>0).mean()))
 for label,sub in [('2020-2025',q.loc[:'2025-12-31']),('2026+',q.loc['2026-01-01':]),('2029+',q.loc['2029-01-01':]),('2030YTD',q.loc['2030-01-01':])]:
  if len(sub)>20: print(' ',label,'n',len(sub),'IC %.6f'%sub.ic.mean(),'ICIR %.6f'%(sub.ic.mean()/sub.ic.std(ddof=1)))
print('coverage %.4f'% (f.notna().sum(axis=1).mean()/len(U)))
print('turnover %.6f'% (f.rank(axis=1,pct=True).diff().abs().sum(axis=1)/2).mean())
print('period',P.index.min(),P.index.max(),'assets',len(U))
f.to_csv('scripts/miner_2_20300822_multihorizon_agreement_signal.csv')

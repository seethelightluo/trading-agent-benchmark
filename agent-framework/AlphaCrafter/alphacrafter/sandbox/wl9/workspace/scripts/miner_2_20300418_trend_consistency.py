import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,2100) for s in U}; frames=[]
for s,d in D.items():
 if d is None or len(d)<200: continue
 x=d.copy(); x['date']=pd.to_datetime(x.date); x=x.set_index('date').sort_index(); r=x.close.pct_change()
 consistency=(r>0).astype(float).rolling(60).mean().shift(1)
 f2=x.close.pct_change(20).shift(1)*(2*consistency-1)
 frames.append(pd.DataFrame({'date':x.index,'symbol':s,'signal':f2.values,'r1':x.close.pct_change(1).shift(-1).values,'r10':x.close.pct_change(10).shift(-10).values,'r20':x.close.pct_change(20).shift(-20).values,'r40':x.close.pct_change(40).shift(-40).values}))
z=pd.concat(frames,ignore_index=True); z[['date','symbol','signal']].dropna().to_csv('scripts/miner_2_20300418_trend_consistency_signal.csv',index=False)
for h in ['r1','r10','r20','r40']:
 vals=[]; ns=[]
 for dt,g in z.groupby('date'):
  q=g[['signal',h]].dropna()
  if len(q)>=8: vals.append(q.signal.corr(q[h],method='spearman')); ns.append(len(q))
 a=np.array(vals); a=a[np.isfinite(a)]
 print(h,'dates',len(a),'avg_n',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/(np.std(a,ddof=1)+1e-12)*np.sqrt(len(a)),6),'hit',round(np.mean(a>0),4))
print('symbols',len(frames),'rows',len(z),'date_range',z.date.min(),z.date.max())

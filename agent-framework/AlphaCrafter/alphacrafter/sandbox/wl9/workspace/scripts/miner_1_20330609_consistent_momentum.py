import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 try: d=get_stock_daily_data(s,days=5000)
 except Exception: d=None
 if d is not None and len(d)>150: frames[s]=d.set_index('date').sort_index()['close'].astype(float)
p=pd.DataFrame(frames).sort_index(); r=p.pct_change()
mom=p/p.shift(60)-1; vol=r.rolling(60).std()*np.sqrt(252); cons=r.rolling(30).mean()/r.rolling(30).std().replace(0,np.nan)
f=-(mom/(vol+0.05) * (1+0.5*np.tanh(cons)))
f.to_csv('scripts/miner_1_20330609_consistent_momentum_signal.csv',index_label='date')
for h in [10,20,40,60]:
 fr=p.shift(-h)/p-1; ics=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
 a=np.array(ics); mu=np.nanmean(a); sd=np.nanstd(a,ddof=1)
 print(h,len(a),round(np.mean(ns),2),f'IC {mu:.6f} ICIR {mu/sd*np.sqrt(252):.6f} hit {np.mean(a>0):.4f}')
fr=p.shift(-60)/p-1; ics=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1])))
for label,lo,hi in [('2020-23','2020','2023-12-31'),('2024-26','2024','2026-12-31'),('2027-29','2027','2029-12-31'),('2030','2030','2030-12-31'),('2031-32','2031','2032-12-31'),('2033','2033','2033-12-31')]:
 a=[v for d,v in ics if str(d)>=lo and str(d)<=hi]; print(label,len(a),round(np.mean(a),6) if a else None)
print('dates',len(p),'assets',len(frames),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))

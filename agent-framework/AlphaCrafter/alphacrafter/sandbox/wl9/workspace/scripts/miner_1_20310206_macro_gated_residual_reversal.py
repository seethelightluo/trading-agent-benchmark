import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d)>150:
  d=d.set_index('date').sort_index(); d.index=pd.DatetimeIndex(d.index).tz_localize(None).astype('datetime64[ns]'); D[s]=d
close=pd.DataFrame({s:d['close'] for s,d in D.items()}).sort_index(); r=close.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']).dt.normalize().astype('datetime64[ns]'); v=v.set_index('date').sort_index()
vc='close' if 'close' in v.columns else ('value' if 'value' in v.columns else v.columns[0]); vix=pd.to_numeric(v[vc],errors='coerce').reindex(close.index).ffill()
cs=r.sub(r.median(axis=1),axis=0); base=-cs.rolling(20,min_periods=15).sum(); vp=vix.rolling(60,min_periods=30).rank(pct=True); sig=(base.mul(0.5+vp,axis=0)).shift(1)
for h in [5,10,20,40,60]:
 vals=[]
 for i in range(len(close.index)-h):
  z=pd.concat([sig.iloc[i],close.iloc[i+h]/close.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append((close.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 q=pd.DataFrame(vals,columns=['date','ic','n']); print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/len(U),4),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if h==60:
  for name,a,b in [('2024-2026','2024-01-01','2026-12-31'),('2027-2029','2027-01-01','2029-12-31'),('2030','2030-01-01','2030-12-31')]:
   q2=q[(q.date>=a)&(q.date<=b)]; print('REG',name,len(q2),round(q2.ic.mean(),6),round(q2.ic.mean()/q2.ic.std(),6))
  print('TURN',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20310206_macro_gated_residual_reversal_signal.csv',index=False)

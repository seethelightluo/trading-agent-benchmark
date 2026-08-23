import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: volume-confirmed short reversal. Reversal is strongest after a large recent move,
# but attenuated when volume confirms the move (continuation risk).
frames={}
for s in U:
    d=None
    for fn in (get_index_daily_data, get_stock_daily_data):
        try: d=fn(s, days=4000)
        except Exception: pass
        if d is not None: break
    if d is not None and len(d)>40:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d
close=pd.DataFrame({s:d['close'] for s,d in frames.items()}); vol=pd.DataFrame({s:d['volume'] for s,d in frames.items()})
ret=close.pct_change()
r3=close/close.shift(3)-1
rv=ret.rolling(15).std()*np.sqrt(15)
# relative volume surprise, clipped for sparse synthetic volume
vr=(vol/vol.rolling(20).median()-1).clip(-2,2)
factor=(-r3/(rv+1e-8))*(1-0.35*vr.clip(lower=0))
# lag is implicit: factor at date t predicts t+1
fwd=close.shift(-1)/close-1
rows=[]
for dt in factor.index:
    x=factor.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(r),'avg_n',round(r.n.mean(),2),'coverage',round(factor.notna().mean().mean(),4))
print('IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4))
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-01')]:
 q=r.loc[a:b,'ic']; print(a,b,'n',len(q),'ic',round(q.mean(),6),'icir',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
# 5/10 day decay for context
for h in [5,10]:
 rr=[]
 fw=close.shift(-h)/close-1
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'n',len(rr),'ic',round(float(np.nanmean(rr)),6),'icir',round(float(np.nanmean(rr)/np.nanstd(rr,ddof=1)),6))
# signal artifact
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20270302_volume_confirmed_reversal_signal.csv',index=False)

import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s, days=6000)
    except Exception as e: print('skip',s,e); x=None
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date').sort_index()
close=pd.concat({s:d['close'] for s,d in D.items()},axis=1).sort_index()
vol=pd.concat({s:d['volume'] for s,d in D.items()},axis=1).sort_index()
vr=np.log(vol.replace(0,np.nan)).rolling(60,min_periods=40).mean()
vs=(np.log(vol.replace(0,np.nan))-vr).shift(1)
r20=close.pct_change(5).shift(1)
f=-(r20*(1+0.5*vs.clip(-2,2))).replace([np.inf,-np.inf],np.nan)
fr=close.shift(-10)/close-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
mu=ic.ic.mean(); sd=ic.ic.std(ddof=1)
print('dates',len(ic),'instruments',len(D),'avgN',ic.n.mean(),'coverage',ic.n.sum()/(len(ic)*len(U)))
print('IC',mu,'ICIR_daily',mu/sd,'hit',(ic.ic>0).mean())
for k in [120,252,756]:
 q=ic.tail(k).ic; print('recent',k,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
for h in [1,3,5,10]:
 ff=close.shift(-h)/close-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'IC',np.nanmean(rr),'obs',len(rr))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20330429_volume_confirmed_short_reversal_signal.csv',index=False)
ic.reset_index().to_csv('scripts/miner_1_20330429_volume_confirmed_short_reversal_ic.csv',index=False)

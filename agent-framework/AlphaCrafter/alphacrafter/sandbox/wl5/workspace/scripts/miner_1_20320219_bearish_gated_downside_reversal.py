import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s, days=5000)
    except Exception: x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.drop_duplicates('date').set_index('date').sort_index(); D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=np.log(p).diff()
shock=-(r.rolling(20).sum())/np.sqrt((r.clip(upper=0)**2).rolling(20).mean())
med=r.rolling(20).sum().median(axis=1); breadth=(-med).clip(lower=0); denom=r.rolling(20).sum().abs().median(axis=1)+1e-12
f=shock*(1+0.5*(breadth/denom).clip(0,1).values[:,None])
rows=[]
for i in range(len(p)-5):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+5]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],len(z),z.f.corr(z.y)))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(a),'mean_n',a.n.mean(),'coverage',a.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(a.ic.mean(),a.ic.mean()/a.ic.std(),(a.ic>0).mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2032-12-31')]:
 q=a.loc[lo:hi].ic; print(lo,hi,len(q),q.mean())
for h in [1,5,10]:
 rr=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8: rr.append(z.f.corr(z.y))
 print('horizon',h,'IC',np.nanmean(rr),'n',len(rr))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20320219_bearish_gated_downside_reversal_signal.csv',index=False)

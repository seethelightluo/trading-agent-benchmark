import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in UNIV:
    d=get_stock_daily_data(s, days=4000)
    if d is not None and len(d)>120:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
        frames[s]=d['close'].astype(float)
prices=pd.concat(frames,axis=1).sort_index()
ret=prices.pct_change()
# Candidate: residual, volatility-normalized intermediate momentum. Signal at t uses through t; IC against t+1..t+10.
cs=ret.median(axis=1)
res=ret.sub(cs,axis=0)
resmom=res.rolling(10,min_periods=8).sum()
vol=ret.rolling(20,min_periods=12).std()*np.sqrt(20)
factor=resmom/vol.replace(0,np.nan)
factor=factor.shift(1) # completed day t-1 at observation date t
fwd=prices.shift(-10)/prices-1
ics=[]; turnovers=[]; cover=[]; ninst=[]
prev=None
for dt in factor.index:
    x=factor.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
        ninst.append(len(z)); cover.append(len(z)/15)
        ranks=x.rank(pct=True)
        if prev is not None: turnovers.append((ranks-prev).abs().mean())
        prev=ranks
arr=np.asarray(ics); arr=arr[np.isfinite(arr)]
print('dates',len(arr),'avg_instruments',np.mean(ninst),'coverage',np.mean(cover),'IC',np.mean(arr),'ICIR_daily',np.mean(arr)/np.std(arr,ddof=1),'hit',np.mean(arr>0),'turnover',np.mean(turnovers))
print('range', prices.index.min(), prices.index.max(), 'factor_valid', factor.notna().sum().sum(), 'fwd_valid', fwd.notna().sum().sum())
for a in np.array_split(arr,3): print('regime',len(a),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1) if len(a)>1 else np.nan)
# decay same factor vs horizons
for h in [1,5,10,20]:
    yy=prices.shift(-h)/prices-1; q=[]
    for dt in factor.index:
      z=pd.concat([factor.loc[dt],yy.loc[dt]],axis=1).dropna()
      if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    q=np.asarray(q);q=q[np.isfinite(q)]
    print('decay',h,len(q),np.mean(q))
# artifact all observations
out=factor.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20291217_residual_momentum10_signal.csv',index=False)

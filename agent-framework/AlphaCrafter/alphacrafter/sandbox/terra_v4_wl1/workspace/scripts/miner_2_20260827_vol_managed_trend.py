import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: volatility-managed trend, trailing return divided by realized vol.
D={}
for s in U:
    x=get_stock_daily_data(s, days=2200)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']).dt.normalize(); x=x.drop_duplicates('date').set_index('date').sort_index()
        D[s]=x['close'].astype(float)
P=pd.concat(D,axis=1).sort_index(); R=P.pct_change()
# forward 1-day return, factor available at t and evaluated t+1
rows=[]
for w in [10,20,40]:
    fac=R.rolling(w,min_periods=max(8,w//2)).sum()/R.rolling(w,min_periods=max(8,w//2)).std()
    for dt in fac.index[:-1]:
        a=fac.loc[dt]; y=R.shift(-1).loc[dt]; z=pd.concat([a,y],axis=1).dropna()
        if len(z)>=8:
            rows.append((w,dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
res=pd.DataFrame(rows,columns=['w','date','n','ic'])
for w,g in res.groupby('w'):
    ic=g.set_index('date').ic.replace([np.inf,-np.inf],np.nan).dropna(); print(f'w={w} dates={len(ic)} avg_n={g.n.mean():.2f} coverage={g.n.sum()/(len(ic)*15):.4f} IC={ic.mean():.8f} ICIR={ic.mean()/ic.std(ddof=1):.8f} hit={(ic>0).mean():.4f}')
    for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31')]:
        q=ic[(ic.index>=lo)&(ic.index<=hi)]; print(' regime',lo,hi,'n',len(q),'icir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan,'mean',q.mean())
    # 5/10 day decay
    for h in [5,10]:
      yy=R.shift(-h).rolling(h).sum() # wrong alignment? sum t+1..t+h via shift(-1).rolling? 
      yy=R.shift(-1).rolling(h).sum().shift(-(h-1))
      vals=[]
      f=(R.rolling(w,min_periods=max(8,w//2)).sum()/R.rolling(w,min_periods=max(8,w//2)).std())
      for dt in f.index:
       z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
       if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
      q=pd.Series(vals).dropna(); print(' horizon',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
print('data_dates',P.index.min(),P.index.max(),'assets',P.shape[1])

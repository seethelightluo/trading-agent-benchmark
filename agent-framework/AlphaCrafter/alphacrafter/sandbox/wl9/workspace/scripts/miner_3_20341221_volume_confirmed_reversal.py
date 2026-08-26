import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
close={}; volume={}
for s,d in D.items():
    if d is not None and len(d)>200:
        q=d.set_index('date'); close[s]=q['close'].astype(float); volume[s]=q['volume'].astype(float)
px=pd.DataFrame(close).sort_index(); volx=pd.DataFrame(volume).reindex(px.index).sort_index(); r=px.pct_change()
# Volume-confirmed medium-horizon reversal: losses that occurred on unusually high activity
ret60=px/px.shift(60)-1
rv60=r.rolling(60,min_periods=45).std()*np.sqrt(60)
vr=volx.rolling(20,min_periods=15).mean()/(volx.rolling(120,min_periods=80).mean()+1e-12)
f=(-ret60/(rv60+1e-6))*np.sqrt(vr.clip(lower=0.25,upper=4.0))
f=f.replace([np.inf,-np.inf],np.nan).shift(1)
f.to_csv('scripts/miner_3_20341221_volume_confirmed_reversal_signal.csv',index_label='date')
for h in [10,20,40,60,80]:
    fr=px.shift(-h)/px-1; vals=[]; ns=[]; dates=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
            if pd.notna(c): vals.append(c); ns.append(len(z)); dates.append(dt)
    x=pd.Series(vals,index=dates).dropna(); ic=x.mean(); icir=ic/x.std(ddof=1)*np.sqrt(252)
    print(f'H={h} dates={len(x)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={icir:.6f} hit={(x>0).mean():.4f}')
valid=f.notna().sum(axis=1); print(f'coverage={valid.mean()/len(U):.6f} instruments={len(U)}')
fr=px.shift(-60)/px-1
for name,a,b in [('2020-23','2020','2023'),('2024-26','2024','2026'),('2027-29','2027','2029'),('2030-32','2030','2032'),('2033-34','2033','2034')]:
    vals=[]
    for dt in f.index:
        if a<=str(dt)[:4]<=b:
            z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
            if len(z)>=8:
                c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
                if pd.notna(c): vals.append(c)
    x=pd.Series(vals).dropna(); print(f'regime={name} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f}')

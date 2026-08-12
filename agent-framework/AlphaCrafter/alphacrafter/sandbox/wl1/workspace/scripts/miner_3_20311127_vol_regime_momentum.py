import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-11-26')
raw={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d[d.date<=cut].sort_values('date'); raw[s]=d.set_index('date')['close']
px=pd.DataFrame(raw).sort_index(); r=np.log(px).diff()
# candidate: medium trend adjusted for recent volatility, with a slow-volatility regime quality tilt
# lag signal one session; evaluate 20d forward return
mom=np.log(px/px.shift(60)); vol=r.rolling(40).std()*np.sqrt(40); shortvol=r.rolling(10).std()*np.sqrt(10)
f=(mom/(vol+1e-9))*(1/(1+shortvol/(vol+1e-9)))
f=f.shift(1)
fwd=np.log(px.shift(-20)/px)
ics=[]; obs=[]; turns=[]
for dt in f.index:
    x=f.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); obs.append(len(z))
    if dt in f.index[1:]:
        a=f.loc[dt- pd.Timedelta(days=1)] if dt-pd.Timedelta(days=1) in f.index else None
        if a is not None:
            q=pd.concat([a,x],axis=1).dropna()
            if len(q)>=8: turns.append((q.iloc[:,0].rank().sub(q.iloc[:,1].rank()).abs().mean()/len(q)))
ics=pd.Series(dict(ics)).dropna(); print('dates',len(ics),'avg_n',np.mean(obs),'coverage',np.mean(obs)/15,'IC20',ics.mean(),'ICIR',ics.mean()/ics.std(),'hit',np.mean(ics>0),'turn',np.nanmean(turns))
for h in [1,5,10,20]:
    vals=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],np.log(px.shift(-h)/px).loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    print('decay',h,np.nanmean(vals))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
    q=ics[(ics.index>=a)&(ics.index<=b)]
    print('regime',a,b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
# artifact, reconstruct full date-index signal csv
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20311127_vol_regime_momentum_signal.csv',index=False)
print('artifact',len(out))

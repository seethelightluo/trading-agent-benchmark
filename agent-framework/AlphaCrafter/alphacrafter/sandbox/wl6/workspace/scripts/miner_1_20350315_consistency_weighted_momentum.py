import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=6000)
    if d is None or len(d)==0: d=get_index_daily_data(s, days=6000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); frames[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(frames).sort_index()
r=np.log(p).diff()
# A simple interpretable signal: medium-term momentum weighted by directional consistency and scaled by risk.
# All inputs end at t-1; forward return starts t and ends t+h-1.
mom=p.pct_change(40)
cons=(r>0).rolling(30,min_periods=20).mean()-0.5
vol=r.rolling(30,min_periods=20).std()
f=(mom*cons/vol).shift(1)
rows=[]
for h in [5,10,20,40]:
    fr=p.shift(-h)/p-1
    ics=[]; n=[]; dates=[]
    for dt in f.index:
        a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(a)>=8:
            ics.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); n.append(len(a)); dates.append(dt)
    z=pd.Series(ics,index=pd.to_datetime(dates)).dropna()
    rows.append((h,len(z),np.mean(n),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)))
    if h==20: selected=(z,n)
print('assets',len(p.columns),list(p.columns)); print('period',p.index.min(),p.index.max(),'rows',len(p))
for x in rows: print('H',x[0],'obs',x[1],'avgN',round(x[2],3),'IC',round(x[3],8),'ICIR',round(x[4],5),'hit',round(x[5],4))
# coverage and turnover on valid cross-sectional ranks
valid=f.notna().sum(axis=1)/len(U)
rank=f.rank(axis=1,pct=True)
turn=rank.diff().abs().mean(axis=1).mean()
print('coverage',valid.mean(),'turnover',turn,'valid_dates',valid.gt(0).sum())
for a,b in [('2020-01-01','2027-12-31'),('2028-01-01','2031-12-31'),('2032-01-01','2035-03-14')]:
 z=selected[0].loc[a:b]; print('regime',a,b,'obs',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
# artifact for deterministic audit
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20350315_consistency_weighted_momentum_signal.csv')

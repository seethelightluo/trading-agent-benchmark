import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def fetch(s):
    d=get_stock_daily_data(s, days=4000)
    if d is None or len(d)<300: d=get_index_daily_data(s, days=4000)
    if d is None: return None
    d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
    return d['close'].astype(float).rename(s)
px=pd.concat([x for x in [fetch(s) for s in U] if x is not None],axis=1).sort_index()
px=px.ffill(limit=3)
r=np.log(px/px.shift(1))
# Cross-sectional dispersion known at t-1; factor is prior 3d relative reversal, activated high dispersion
csdisp=r.shift(1).std(axis=1)
# rolling percentile computed using observations through t-1
thr=csdisp.shift(1).rolling(120,min_periods=60).quantile(.70)
active=(csdisp.shift(1)>thr).astype(float)
prior3=np.log(px.shift(1)/px.shift(4))
base=-(prior3.sub(prior3.median(axis=1),axis=0))
f=base.mul(active,axis=0)
# normalize not required for IC; zero rows omitted
print('loaded',len(px),'dates',px.index.min(),px.index.max(),'names',len(px.columns))
for h in [1,3,5,10]:
    fr=np.log(px.shift(-h)/px)
    vals=[]; dates=[]; cov=[]
    for dt in px.index:
        a=f.loc[dt]; b=fr.loc[dt]
        z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1 and active.loc[dt]>0:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); cov.append(len(z)/len(U))
    q=pd.Series(vals,index=dates).dropna()
    print('H',h,'n',len(q),'meanIC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit', (q>0).mean(),'coverage',np.mean(cov))
    for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2029')]:
        x=q[(q.index>=a)&(q.index<=b+'-12-31')]
        if len(x): print(a,b,round(x.mean(),6),round(x.mean()/x.std(ddof=1),4),len(x))
# rank turnover on active dates
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
    if active.iloc[i]>0 and active.iloc[i-1]>0:
        z=pd.concat([ranks.iloc[i],ranks.iloc[i-1]],axis=1).dropna(); turns.append(np.mean(abs(z.iloc[:,0]-z.iloc[:,1])))
print('turnover_proxy',np.mean(turns),'active_frac',active.mean(),'coverage',f.notna().mean().mean())
f.to_csv('scripts/miner_2_20290531_dispersion_conditioned_relative_reversal_signal.csv',index_label='date')

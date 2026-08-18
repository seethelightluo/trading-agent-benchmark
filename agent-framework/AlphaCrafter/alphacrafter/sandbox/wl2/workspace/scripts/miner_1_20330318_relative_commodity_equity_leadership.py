import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,5000)
            if x is not None and len(x)>100:return x
        except Exception: pass
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.set_index(pd.to_datetime(x.date)).close.astype(float) for s,x in D.items()}).sort_index().groupby(level=0).last().ffill()
R=C.pct_change(); med=R.median(axis=1)
# Relative commodity/equity leadership: residual medium-term trend, activated only
# when lagged cyclical commodity leadership is neither weak nor euphoric.
cyc=['COPPER','WTI']; eq=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']
cyc=[s for s in cyc if s in R]; eq=[s for s in eq if s in R]
lead=(R[cyc].rolling(20,min_periods=12).mean().mean(axis=1)-R[eq].rolling(20,min_periods=12).mean().mean(axis=1)).shift(1)
gate=((lead>=-0.001)&(lead<=0.004)).astype(float)
trend=R.rolling(20,min_periods=15).sum().shift(1).sub(R.rolling(20,min_periods=15).sum().shift(1).median(axis=1),axis=0)
vol=R.rolling(40,min_periods=20).std().shift(1)
f=(trend/vol).mul(gate,axis=0).replace([np.inf,-np.inf],np.nan)
print('assets',len(D),'dates',len(C),'gate_dates',int(gate.sum()),'active_coverage',round(f.notna().sum(axis=1).replace(0,np.nan).mean()/len(D),4))
for h in [1,3,5,10]:
    fr=R.rolling(h).sum().shift(-h); vals=[]; ds=[]
    for d in f.index:
        z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1:
            vals.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank())); ds.append(d)
    a=pd.Series(vals,index=ds).dropna(); print('H',h,'dates',len(a),'nassets>=8','15','IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(),6),'hit',round((a>0).mean(),4))
    if h==1 and len(a):
        k=len(a)//2; print('early_late',round(a.iloc[:k].mean(),6),round(a.iloc[k:].mean(),6))
f.to_csv('scripts/miner_1_20330318_relative_commodity_equity_leadership_signal.csv',index_label='date')

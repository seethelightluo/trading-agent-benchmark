import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
    d=None
    for f in (get_index_daily_data,get_stock_daily_data):
        try: d=f(s,4200)
        except Exception: d=None
        if d is not None: break
    if d is not None and len(d)>100:
        d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in P.items()}); r=cl.pct_change(); vol=r.rolling(40,min_periods=20).std()
t20=(cl/cl.shift(20)-1)/(vol*np.sqrt(20)); t60=(cl/cl.shift(60)-1)/(vol*np.sqrt(60))
# Lagged continuous breadth confidence: signed distance of positive-return breadth from 50%.
b=((r.rolling(20).sum()>0).mean(axis=1)).shift(1)
for gamma in [0.25,0.5,1.0,2.0]:
    conf=np.sign(2*b-1)*np.abs(2*b-1)**gamma
    sig=(0.4*t20+0.6*t60).mul(conf,axis=0).shift(1)
    vals=[]; ns=[]
    for dt in sig.index:
        y=cl.shift(-20).loc[dt]/cl.shift(-1).loc[dt]-1
        z=pd.concat([sig.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)>=8:
            q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
            if np.isfinite(q): vals.append(q); ns.append(len(z))
    x=pd.Series(vals)
    print('gamma',gamma,'dates',len(x),'meanIC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'avgN',round(np.mean(ns),2),'recent756ICIR',round(x.tail(756).mean()/x.tail(756).std(ddof=1),6))
# persist artifact for gamma=0.25 candidate for possible admission
gamma=.25; conf=np.sign(2*b-1)*np.abs(2*b-1)**gamma
sig=(.4*t20+.6*t60).mul(conf,axis=0).shift(1)
sig.to_csv('scripts/miner_2_20351001_breadth_confidence_power_signal.csv')
print('universe',len(cl.columns),'calendar_dates',len(cl),'coverage',round(sig.notna().mean().mean(),4))

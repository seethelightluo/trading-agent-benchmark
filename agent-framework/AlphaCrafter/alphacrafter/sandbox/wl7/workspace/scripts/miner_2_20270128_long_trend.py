import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2027-01-27')
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,3000)
            if x is not None and len(x):
                x=x.copy(); x.date=pd.to_datetime(x.date).dt.normalize()
                return x.drop_duplicates('date').set_index('date').sort_index().loc[:CUT]
        except Exception: pass
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.concat({s:d.close for s,d in D.items()},axis=1)
# 60d trend with 20d volatility, lagged one day
F=(C.pct_change(60)/C.pct_change().rolling(20).std()).shift(1)
FR=pd.concat({s:D[s].close.shift(-1)/D[s].close-1 for s in D},axis=1)
def run(lo=None,hi=None):
    vals=[]; ns=[]
    for dt in F.index:
        if lo and dt<pd.Timestamp(lo): continue
        if hi and dt>pd.Timestamp(hi): continue
        q=pd.concat([F.loc[dt],FR.loc[dt]],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,0].nunique()>1:
            vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q))
    a=pd.Series(vals)
    return len(a),round(np.mean(ns),2),round(a.mean(),5),round(a.mean()/a.std(ddof=1)*np.sqrt(252),4),round((a>0).mean(),4)
print('assets',len(D),'dates',F.index.min().date(),F.index.max().date(),'overall',run(),'coverage',round(F.notna().mean().mean(),4))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-27')]: print(lo[:4]+'-'+hi[:4],run(lo,hi))
r=F.rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean(axis=1).mean(),4))
# decay horizons
for h in [5,10,20]:
    T=pd.concat({s:D[s].close.shift(-h)/D[s].close-1 for s in D},axis=1); vals=[]
    for dt in F.index:
        q=pd.concat([F.loc[dt],T.loc[dt]],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
    print('decay',h,len(vals),round(np.nanmean(vals),5))
r.to_csv('scripts/miner_2_20270128_long_trend_signal.csv')

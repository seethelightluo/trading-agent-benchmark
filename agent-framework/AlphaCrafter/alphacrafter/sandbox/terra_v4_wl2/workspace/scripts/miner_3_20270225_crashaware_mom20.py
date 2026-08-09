import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            x=fn(s,days=5000)
            if x is not None and len(x): return x
        except Exception: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index()
r=px.pct_change(); mom=r.rolling(20,min_periods=15).sum(); vol=r.rolling(20,min_periods=15).std(); down=r.where(r<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
# interpretable crash-aware momentum: return / total volatility, with extra penalty for downside volatility
f=(mom/(vol+1e-8))/(1+down/(vol+1e-8)); f=f.sub(f.mean(axis=1),axis=0)
for h in [1,5,10]:
    fr=px.shift(-h)/px-1; vals=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    a=np.array(vals); print('H',h,'dates',len(a),'avg_n',round(float(np.mean(ns)),2),'IC',float(np.mean(a)),'ICIR',float(np.mean(a)/np.std(a,ddof=1)),'hit',float(np.mean(a>0)))
    for lab,lo,hi in [('2020-22','2020','2022-12-31'),('2023-24','2023','2024-12-31'),('2025-26','2025','2026-12-31'),('2027','2027','2027-02-25')]:
        q=[x for d,x in zip(f.index,vals) if str(d)>=lo and str(d)<=hi] if False else []
        # recompute regime from date-aligned observations
        q=[]
        for dt in f.index:
            z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
            if len(z)>=8 and z.iloc[:,0].nunique()>1 and str(dt)>=lo and str(dt)<=hi:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
        print('REG',h,lab,len(q),float(np.mean(q)) if q else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_3_20270225_crashaware_mom20.csv',index=False)
print('dates',len(f),'instruments',len(U),'coverage',float(f.notna().sum().sum()/(len(f)*len(U))))
print('turnover',float((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.05).mean()))

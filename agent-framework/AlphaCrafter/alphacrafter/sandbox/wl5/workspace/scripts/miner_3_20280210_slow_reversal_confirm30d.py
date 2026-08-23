import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
E=pd.Timestamp('2028-02-09')
def load(s):
    return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date')
d={s:load(s) for s in U}; c=pd.concat({s:x.close for s,x in d.items()},axis=1).sort_index(); r=c.pct_change()
ret=c/c.shift(60)-1; vol=r.rolling(60,min_periods=40).std()*np.sqrt(60)
# 60d risk-adjusted reversal, activated only when the independent 30d trend is positive.
confirm=(c/c.shift(30)-1>0).astype(float)
f=(-(ret/(vol+1e-12))*confirm).rank(axis=1,pct=True).where(ret.notna())
y=c.pct_change(10).shift(-10); a=[]; ns=[]; ds=[]
for dt in f.index:
    z=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
    if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
        a.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
a=np.asarray(a); ds=np.asarray(ds)
print('factor=slow_reversal_60d_confirm30d horizon=10 cutoff',E.date())
print('dates',len(a),'avgN',round(np.mean(ns),3),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2028)]:
 q=a[[lo<=pd.Timestamp(x).year<=hi for x in ds]]
 print('regime',lo,hi,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'hit',round((q>0).mean(),4) if len(q) else None)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_3_20280210_slow_reversal_confirm30d_signal.csv',index=False)

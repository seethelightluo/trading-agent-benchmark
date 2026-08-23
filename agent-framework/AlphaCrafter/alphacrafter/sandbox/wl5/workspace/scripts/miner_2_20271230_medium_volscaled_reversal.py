import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2027-12-15')
def ld(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date')
c=pd.concat({s:ld(s).close for s in U},axis=1).sort_index(); r=c.pct_change();
# Medium-horizon reversal, scaled by recent risk: unusually negative 20d return is favored.
f=-(c/c.shift(20)-1)/(r.rolling(30,min_periods=20).std()*np.sqrt(20)); f=f.sub(f.median(axis=1),axis=0); y=c.pct_change(10).shift(-10)
a=[];ns=[];ds=[]
for dt in f.index:
 z=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:a.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
a=np.array(a); print('factor medium_volscaled_reversal_20_30'); print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4));
for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
 q=a[[lo<=x.year<=hi for x in ds]];print('regime',lo,hi,'dates',len(q),'IC',round(q.mean(),6) if len(q) else None,'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4)); f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_2_20271230_medium_volscaled_reversal_signal.csv',index=False)

import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-02-25')
frames={}
for s in U:
    d=get_stock_daily_data(s,2200)
    if d is None: d=get_index_daily_data(s,2200)
    if d is not None and len(d):
        d=d[d.date<=cut].copy(); d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d
prices=pd.DataFrame({s:d.close for s,d in frames.items()}).sort_index()
r=prices.pct_change()
# Candidate: medium-term momentum penalized by downside risk, with skip-2d to avoid very short reversal
ret=prices.shift(2)/prices.shift(22)-1
neg=r.where(r<0,0.0)
down=neg.rolling(20,min_periods=15).std()*np.sqrt(252)
f=ret/down.replace(0,np.nan)
# winsorize cross section to reduce crypto influence, still interpretable
f=f.clip(lower=f.quantile(.05,axis=1),upper=f.quantile(.95,axis=1),axis=0)
for h in [1,5,10]:
    vals=[]; cov=[]; turns=[]
    for i,dt in enumerate(prices.index):
        if i+h>=len(prices): continue
        x=f.loc[dt]; y=prices.iloc[i+h]/prices.iloc[i]-1
        z=pd.concat([x,y],axis=1).dropna();
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); cov.append(len(z)/15)
    a=pd.Series(vals).dropna(); print('H',h,'dates',len(a),'avgN',round(np.mean(np.array(cov)*15),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round((a>0).mean(),4),'coverage',round(np.mean(cov),4))
# date/instrument signal artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out['date']=out.date.dt.strftime('%Y-%m-%d'); out.to_csv('../persistent/factor_signals_miner_2_20270225_downside_momentum20.csv',index=False)
print('artifact rows',len(out),'dates',out.date.nunique(),'symbols',out.symbol.nunique())
for yr in [2020,2021,2022,2023,2024,2025,2026,2027]:
    mask=[prices.index[i].year==yr for i in range(len(prices.index))]
    # recompute 10d yearly
    a=[]
    for i,dt in enumerate(prices.index[:-10]):
      if dt.year!=yr: continue
      z=pd.concat([f.loc[dt],prices.iloc[i+10]/prices.iloc[i]-1],axis=1).dropna()
      if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    if len(a)>3: print('YR',yr,'n',len(a),'IC',round(np.mean(a),5),'ICIR',round(np.mean(a)/np.std(a,ddof=1),5))

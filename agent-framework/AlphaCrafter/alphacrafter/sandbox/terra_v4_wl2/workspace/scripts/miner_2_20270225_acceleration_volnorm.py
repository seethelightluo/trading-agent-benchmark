import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
 px[a]=d['close']
prices=pd.concat(px,axis=1).sort_index()
# factor: medium-term acceleration, normalized by 20d realized vol to make comparable
r20=prices/prices.shift(20)-1
r60=prices/prices.shift(60)-1
ret=prices.pct_change()
vol=ret.rolling(20).std()*np.sqrt(20)
f=(r20-r60)/vol.replace(0,np.nan)
f=f.replace([np.inf,-np.inf],np.nan)
fwd=prices.shift(-1)/prices-1
ics=[]; rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ic=spearmanr(x[ok],y[ok]).statistic
  ics.append(ic); rows.append((dt,ic,ok.sum()))
ics=np.array(ics)
print('dates',len(ics),'avg_n',np.mean([r[2] for r in rows]),'IC',np.nanmean(ics),'ICIR',np.nanmean(ics)/np.nanstd(ics,ddof=1),'hit',np.mean(ics>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
 z=[r[1] for r in rows if str(r[0])[:10]>=lo and str(r[0])[:10]<=hi]
 if z: print(lo, len(z), np.mean(z),np.mean(z)/np.std(z,ddof=1),np.mean(np.array(z)>0))
# turnover rank proxy
rank=f.rank(axis=1,pct=True); delta=(rank-rank.shift(1)).abs().mean(axis=1).dropna(); print('turnover',delta.mean(),'coverage',f.notna().mean().mean())
out=f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal'); out.to_csv('../persistent/factor_signals_miner_2_20270225_acceleration_volnorm.csv',index=False)

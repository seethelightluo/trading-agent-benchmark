import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv'); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:'2029-11-28']; R=P.pct_change(); v20=R.rolling(20,min_periods=15).std(); v60=R.rolling(60,min_periods=40).std()
# Trend signal with volatility compression tilt; simple interpretable cross-asset factor
F=(P.pct_change(20)/(v20+1e-12))*(1-v20/(v60+1e-12)); F=F.replace([np.inf,-np.inf],np.nan)
print('candidate=vol_compression_trend universe',len(U),'dates',P.index.min().date(),P.index.max().date())
for h in [5,10,20,40]:
 fr=P.shift(-h).div(P)-1; vals=[]; dates=[]; ns=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(a))
 x=np.asarray(vals); ds=pd.DatetimeIndex(dates); sd=x.std(ddof=1)
 print('H',h,'dates',len(x),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'hit',round(np.mean(x>0),4))
 for label,lo,hi in [('early','2020-01-01','2022-12-31'),('mid','2023-01-01','2026-07-15'),('online','2026-07-16','2029-11-28'),('recent','2028-11-15','2029-11-28')]:
  z=x[(ds>=lo)&(ds<=hi)]; print(' ',label,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'valid_coverage',round(F.notna().mean().mean(),6))
F.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20291129_vol_compression_trend_signal.csv',index=False)

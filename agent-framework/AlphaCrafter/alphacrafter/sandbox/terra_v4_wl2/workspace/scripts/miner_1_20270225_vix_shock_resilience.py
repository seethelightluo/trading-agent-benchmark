import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
px={s:pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
prices=pd.DataFrame(px).sort_index(); rets=prices.pct_change(); vr=vix.reindex(prices.index).pct_change()
shock=(vr>vr.rolling(120,min_periods=60).quantile(.75)).astype(float)
f=rets.mul(shock,axis=0).rolling(60,min_periods=20).mean()/rets.rolling(60,min_periods=20).std()
ics=[]; turns=[]; counts=[]; dates=[]; prev=None
for d in prices.index:
 x=f.loc[d]; y=rets.shift(-1).loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); counts.append(ok.sum());dates.append(d)
  r=x.rank(pct=True); turns.append(np.nan if prev is None else np.mean(abs(r-prev))); prev=r
arr=np.array(ics); print('idea=VIX-shock resilience, dates',len(arr),'avg_names',np.mean(counts),'coverage',np.mean(counts)/15,'IC',np.nanmean(arr),'ICIR',np.nanmean(arr)/np.nanstd(arr,ddof=1),'hit',np.mean(arr>0),'turnover',np.nanmean(turns))
for label,mask in [('2020-22',(pd.Series(dates).dt.year<=2022)),('2023-24',pd.Series(dates).dt.year.isin([2023,2024])),('2025+', (pd.Series(dates).dt.year>=2025))]:
 z=arr[mask.values];print(label,'dates',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'hit',np.mean(z>0))
for h in [5,10]:
 z=[];n=[]
 for d in prices.index:
  x=f.loc[d]; y=prices.pct_change(h).shift(-h).loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:z.append(spearmanr(x[ok],y[ok]).statistic);n.append(ok.sum())
 z=np.array(z);print('h',h,'dates',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'names',np.mean(n))
f.to_csv('../persistent/factor_signals_miner_1_20270225_vix_shock_resilience.csv',index_label='date')

import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv'); d['date']=pd.to_datetime(d['date']); px[a]=d.set_index('date')['close'].astype(float)
prices=pd.DataFrame(px).sort_index(); rets=prices.pct_change()
med=rets.rolling(10).sum().median(axis=1); raw=-(rets.rolling(10).sum().sub(med,axis=0)); vol=rets.rolling(20).std()*np.sqrt(10)
sig=(raw/vol).replace([np.inf,-np.inf],np.nan); disp=rets.rolling(10).sum().std(axis=1); q=disp.rolling(252,min_periods=60).rank(pct=True)
factor=sig.mul(0.5+q,axis=0)
ics={h:[] for h in [5,10,20,40]}; dates={h:[] for h in ics}; cov=[]; turnovers=[]; prev=None
for t in range(1,len(prices)-40):
 f=factor.iloc[t-1]
 for h in ics:
  z=pd.concat([f,prices.pct_change(h).shift(-h).iloc[t]],axis=1).dropna()
  if len(z)>=8: ics[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates[h].append(prices.index[t])
 z=f.dropna(); cov.append(len(z)/15); r=z.rank(); turnovers.append((r-prev).abs().mean()/15 if prev is not None else np.nan); prev=r
print('assets',len(assets),'dates',len(prices),'range',prices.index.min(),prices.index.max())
for h in ics:
 x=np.array(ics[h]); print(h,'IC',np.nanmean(x),'ICIR',np.nanmean(x)/np.nanstd(x,ddof=1),'hit',np.mean(x>0),'n',len(x))
print('coverage',np.nanmean(cov),'turnover',np.nanmean(turnovers))
for lo,hi in [(2020,2023),(2024,2026),(2027,2030)]:
 x=np.array([v for v,d in zip(ics[10],dates[10]) if lo<=d.year<=hi]); print(lo,hi,'10d',len(x),np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1))
pd.DataFrame(factor).to_csv('scripts/miner_1_20300307_dispersion_gated_reversal_signal.csv')

import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index().loc[:'2034-09-01']; r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# Breadth-stress reversal: lagged 5d reversal amplified only when breadth is unusually narrow.
breadth=(r.rolling(20).mean()>0).mean(axis=1)
stress=(0.5-breadth).clip(lower=0).rolling(60).mean()
vstress=((vix-vix.rolling(120).mean())/vix.rolling(120).std()).clip(lower=0,upper=3)
f=(-r.rolling(5).sum()).shift(1).mul((1+2*stress+0.25* vstress).shift(1),axis=0)
f=f.replace([np.inf,-np.inf],np.nan)
fwd=p.shift(-10)/p-1
ics=[]; dates=[]; ns=[]
for d in p.index:
 x=f.loc[d]; y=fwd.loc[d]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); dates.append(d); ns.append(ok.sum())
ic=np.array(ics); print('dates',len(ic),'avgN',np.mean(ns),'coverage',np.mean(ns)/15)
for label,z in [('full',ic),('120',ic[-120:]),('252',ic[-252:]),('756',ic[-756:])]:
 print(label,'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1),'hit',np.mean(z>0))
print('turnover',np.mean(np.sign(f).diff().abs().sum(axis=1)>0))
print('period',dates[0],dates[-1])

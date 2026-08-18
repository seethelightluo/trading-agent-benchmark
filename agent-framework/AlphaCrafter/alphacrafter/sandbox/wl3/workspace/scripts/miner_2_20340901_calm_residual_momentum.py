import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:'2034-09-01']; r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
# Relative momentum residual: each asset's 20d return less cross-sectional median, weighted toward calm/stable regimes.
raw=r.rolling(20).sum(); resid=raw.sub(raw.median(axis=1),axis=0)
vol=r.rolling(20).std(); calm=(1/(1+vol)).clip(upper=30)
reg=(1/(1+((vix-vix.rolling(120).mean())/vix.rolling(120).std()).clip(lower=0))).shift(1)
f=resid.shift(1)*calm.shift(1).mul(reg,axis=0)
y=p.shift(-10)/p-1
ics=[]; ns=[]
for d in p.index:
 ok=f.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8: ics.append(spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic); ns.append(ok.sum())
ic=np.array(ics); print('dates',len(ic),'avgN',np.mean(ns),'coverage',np.mean(ns)/15)
for k,z in [('full',ic),('120',ic[-120:]),('252',ic[-252:]),('756',ic[-756:])]: print(k,'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1),'hit',np.mean(z>0))
print('period',p.index[0],p.index[-1])
# rank turnover
print('turnover',np.mean((f.rank(axis=1,pct=True).diff().abs().sum(axis=1)>0)))

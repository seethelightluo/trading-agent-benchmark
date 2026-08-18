import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2028-08-29')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); d=d[d.date<=cutoff].sort_values('date').set_index('date'); px[s]=d.close.astype(float)
X=pd.DataFrame(px).sort_index(); r=X.pct_change(); vol=r.rolling(20,min_periods=15).std().shift(1)
raw=(-r/(vol+1e-12)).shift(1); f=raw.rolling(3,min_periods=3).mean(); fr=X.shift(-10)/X-1
ics=[]; cov=[]
for dt in X.index:
 a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna(); n=int(ok.sum())
 if n>=8: ics.append((dt,spearmanr(a[ok],b[ok]).statistic));cov.append(n/15)
v=pd.Series(dict(ics)); ranks=f.rank(axis=1,pct=True); turns=(ranks-ranks.shift(1)).abs().mean(axis=1)
print('dates',len(v),'mean_instruments',np.mean(np.array(cov)*15),'coverage',np.mean(cov));print('IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0),'turnover_proxy',turns.loc[v.index].mean());print('last',v.index.max())
for label,mask in [('2027',v.index.year==2027),('2028',v.index.year==2028),('recent60',np.arange(len(v))>=len(v)-60),('recent120',np.arange(len(v))>=len(v)-120),('recent252',np.arange(len(v))>=len(v)-252)]:
 z=v[mask]; print(label,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan,np.mean(z>0))
v.to_csv('scripts/miner_2_20280829_smoothed_volnorm_reversal_ic.csv',header=['ic']); f.loc[:cutoff].to_csv('scripts/miner_2_20280829_smoothed_volnorm_reversal_signal.csv')

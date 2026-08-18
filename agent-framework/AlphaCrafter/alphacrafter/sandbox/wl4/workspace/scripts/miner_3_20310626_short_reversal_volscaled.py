import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d.close
p=pd.DataFrame(px).sort_index()
r=p.pct_change()
# candidate: lagged 3d residual reversal, scaled by trailing 20d vol; conditioned on ordinary vol not extreme
raw=r.rolling(3).sum()
vol=r.rolling(20).std()
cs=raw.sub(raw.median(axis=1),axis=0)
f=-cs/(vol+1e-8)
# use only values where vol is below cross-sectional 75th percentile (soft breadth gate)
gate=vol.le(vol.quantile(.75,axis=1),axis=0)
f=f.where(gate)
# lag 1 completed day
f=f.shift(1)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1
 vals=[]
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(vals).dropna(); print('H',h,'dates',len(x),'avgN',round(np.nanmean([len(pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()) for d in f.index if len(pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna())>=8]),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
# turnover rank signal, coverage
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('coverage',f.notna().sum().sum()/p.notna().sum().sum(),'turnover',turn,'period',p.index.min().date(),p.index.max().date())
for n in [365,730,1095]:
 x=[]
 for dt in f.index[-n:]:
  a=f.loc[dt]; b=(p.shift(-10)/p-1).loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=pd.Series(x).dropna();print('recent',n,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'dates',len(x))

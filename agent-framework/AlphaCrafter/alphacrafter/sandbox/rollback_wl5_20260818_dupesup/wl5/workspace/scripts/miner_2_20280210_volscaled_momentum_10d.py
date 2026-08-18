import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(p)
 x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date')
 D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# interpretable: 10d price trend normalized by trailing 20d realized volatility
fac=(p.pct_change(10))/(r.rolling(20).std()*np.sqrt(10))
fwd=p.shift(-1)/p-1
ics=[]; turnovers=[]; valid=[]
for dt in fac.index:
 a=fac.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); valid.append(len(z))
  turnovers.append((a.rank(pct=True)-fac.shift(1).loc[dt].rank(pct=True)).abs().mean())
ics=np.array(ics); turnovers=np.array(turnovers)
print({'dates':len(ics),'instruments_mean':round(float(np.mean(valid)),2),'coverage':round(float(np.mean(valid)/15),4),'mean_ic':round(float(np.nanmean(ics)),6),'abs_ic':round(float(abs(np.nanmean(ics))),6),'icir':round(float(np.nanmean(ics)/np.nanstd(ics,ddof=1)),6),'hit':round(float(np.mean(ics>0)),4),'turnover':round(float(np.nanmean(turnovers)),4)})
for h in [1,5,10,20]:
 q=p.shift(-h)/p-1; vals=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],q.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(h,round(float(np.nanmean(vals)),6),len(vals))
print('period',fac.index.min().date(),fac.index.max().date())

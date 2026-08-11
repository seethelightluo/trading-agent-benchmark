import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.split('/')[-1][:-4]
 if s in U:
  data[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15']
close=pd.DataFrame({s:d.close for s,d in data.items()}); ret=close.pct_change()
vol20=ret.rolling(20,min_periods=10).std(); vol60=ret.rolling(60,min_periods=20).std()
fwd=ret.shift(-1)
variants={'highvol_scaled_reversal':-ret*vol20/vol60,'lowvol_scaled_reversal':-ret*vol60/vol20,'volshock_reversal':-ret*(vol20/vol60-1)}
for name,x in variants.items():
 vals=[]; cov=[]; turnovers=[]; prev=None
 for dt in x.index:
  z=pd.concat([x.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); cov.append(len(z)/15)
   r=x.loc[dt].rank(pct=True).dropna(); turnovers.append(np.nan if prev is None else np.mean(abs(r-prev.reindex(r.index).fillna(.5)))); prev=r
 ic=np.array(vals); icir=ic.mean()/ic.std(ddof=1)*np.sqrt(252)
 print(name,'dates',len(ic),'avg_n',np.mean(cov)*15,'IC',ic.mean(),'ICIR',icir,'hit',np.mean(ic>0),'turn',np.nanmean(turnovers),'recent250',ic[-250:].mean())
 for h in [5,10]:
  yy=close.shift(-h)/close-1;q=[]
  for dt in x.index:
   z=pd.concat([x.loc[dt],yy.loc[dt]],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  print(' decay',h,np.mean(q))

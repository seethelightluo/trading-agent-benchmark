import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for fn in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(fn)[:-4]
 if s in keep:
  q=pd.read_csv(fn); q.date=pd.to_datetime(q.date); d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2032-11-10']; r=px.pct_change()
# Volatility-scaled 5-day reversal: recent losers receive larger scores, normalized by trailing 20d volatility.
# Lag one completed day; interpretable and high coverage.
vol=r.rolling(20,min_periods=15).std().replace(0,np.nan)
f=(-px.pct_change(5)/vol).shift(1)
print('candidate=vol_scaled_reversal_5_20; universe=15; through=2032-11-10')
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
for label,a,b in [('2020-23','2020','2023-12-31'),('2024-27','2024','2027-12-31'),('2028-30','2028','2030-12-31'),('2031-32','2031','2032-11-10')]:
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],(px.shift(-1)/px-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.array(vals); print('REG',label,len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/(x.std(ddof=1)+1e-12),6))
print('coverage',round(f.notna().mean().mean(),4),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4),'meanvalid',round(f.notna().sum(axis=1).replace(0,np.nan).mean(),2))
print('library correlation audit: pending efficacy')

import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
keep=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for fn in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(fn)[:-4]
 if s in keep:
  q=pd.read_csv(fn);q.date=pd.to_datetime(q.date);d[s]=q.set_index('date').close
px=pd.DataFrame(d).sort_index().loc[:'2032-11-10'];r=px.pct_change()
# Low-volatility factor: inverse trailing 20-session realized volatility, lagged one session.
f=(1/r.rolling(20,min_periods=15).std()).shift(1)
print('candidate=inverse_realized_volatility_20; universe=15; through=2032-11-10')
for h in [1,5,10,20]:
 vals=[];ns=[];fr=px.shift(-h)/px-1
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals);print('H',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),4),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean().mean(),4),'meanvalid',round(f.notna().sum(axis=1).replace(0,np.nan).mean(),2))
print('library correlation audit: pending efficacy')

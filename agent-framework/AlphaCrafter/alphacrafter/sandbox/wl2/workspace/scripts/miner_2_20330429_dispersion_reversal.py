import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-04-29')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in assets}
p=pd.DataFrame(px).sort_index();p=p[p.index<=cut];r=p.pct_change(); cs5=r.rolling(5).mean(); dispersion=r.std(axis=1).rolling(20).mean(); high=dispersion>dispersion.rolling(252).quantile(.7)
# lagged 5d cross-sectional residual reversal active in high dispersion, scaled by 20d vol
raw=-(p.pct_change(5).sub(p.pct_change(5).median(axis=1),axis=0)); f=raw/r.rolling(20).std();f=f.where(high, np.nan)
for h in [1,3,5,10]:
 fr=p.pct_change(h).shift(-h);ics=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(ics);print(h,'dates',len(a),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
for s,e in [('2026-01-01','2029-12-31'),('2030-01-01','2033-04-29')]:
 a=[]
 for dt in f.index:
  if not(pd.Timestamp(s)<=dt<=pd.Timestamp(e)):continue
  z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(s,len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())

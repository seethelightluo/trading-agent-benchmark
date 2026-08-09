"""Candidate: residual upside participation acceleration, completed data through 2032-11-10."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
AS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; START=pd.Timestamp('2026-07-16'); END=pd.Timestamp('2032-11-10')
D={a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index() for a in AS};idx=pd.DatetimeIndex(sorted(set.intersection(*[set(x.loc[(x.index>=START)&(x.index<=END)].index) for x in D.values()])))
c=pd.DataFrame({a:D[a].reindex(idx).close for a in AS});r=c.pct_change();common=r.median(axis=1);beta=r.rolling(60,min_periods=45).cov(common).div(common.rolling(60,min_periods=45).var(),axis=0);res=r-beta.mul(common,axis=0)
s=res.gt(0).rolling(20,min_periods=15).mean()-res.gt(0).rolling(60,min_periods=45).mean()
print('FACTOR residual_upside_participation_acceleration_20_60obs cutoff',END.date(),'assets',len(AS),'dates',len(idx));print('coverage',int(s.notna().sum().sum()),'/',s.size,round(s.notna().mean().mean(),6))
out={}
for h in [1,5,10,20]:
 f=c.shift(-h).div(c).sub(1);vals=[];ns=[];ds=[]
 for d in idx:
  x=pd.concat([s.loc[d],f.loc[d]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:
   v=spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);ns.append(len(x));ds.append(d)
 vals=np.array(vals);ds=pd.DatetimeIndex(ds);ns=np.array(ns);out[h]=(vals,ds)
 print('H',h,'dates',len(vals),'IC',round(vals.mean(),6),'ICIR',round(vals.mean()/vals.std(ddof=1),6),'hit',round((vals>0).mean(),6),'mean_n',round(ns.mean(),3),'min_n',int(ns.min()),'PASS',abs(vals.mean())>=.007 and abs(vals.mean()/vals.std(ddof=1))>=.084)
vals,ds=out[20]
for n,lo,hi in [('2026_2029','2026-07-16','2029-12-31'),('2030_plus','2030-01-01',END),('recent_12m','2031-11-11',END)]:
 x=vals[(ds>=lo)&(ds<=hi)];print('REGIME',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=s.rank(axis=1,pct=True);print('turnover',round((rk-rk.shift()).abs().stack().mean(),6),'median_iqr',round((s.quantile(.75,axis=1)-s.quantile(.25,axis=1)).median(),6))

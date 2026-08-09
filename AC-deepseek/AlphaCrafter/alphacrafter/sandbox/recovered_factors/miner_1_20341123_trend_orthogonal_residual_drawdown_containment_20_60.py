# Single-idea validation: trend-orthogonal residual drawdown containment (20/60 sessions).
# Higher score means an asset has contained its idiosyncratic (cross-asset-beta-adjusted)
# 20-session drawdown relative to its own residual volatility.  The signal is made
# cross-sectionally orthogonal to 20-session risk-adjusted price trend.
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 p[a]=pd.to_numeric(d['close'],errors='coerce').replace(0,np.nan)
c=pd.DataFrame(p).sort_index(); r=c.pct_change(fill_method=None)
# Each beta is estimated strictly from the preceding 60 completed returns.
mkt=r.median(axis=1,skipna=True)
beta=r.rolling(60,min_periods=45).cov(mkt).div(mkt.rolling(60,min_periods=45).var(),axis=0)
resid=r.sub(beta.mul(mkt,axis=0))
# Compound residual daily returns into a 20-session idiosyncratic wealth path; measure
# current shortfall from its rolling high and scale by ex-ante residual volatility.
reswealth=(1+resid.fillna(0)).cumprod()
resdd=reswealth/reswealth.rolling(20,min_periods=16).max()-1
idvol=resid.rolling(60,min_periods=45).std()
raw=resdd.div(idvol.replace(0,np.nan)) # nearer zero is stronger containment
trend=c.pct_change(20).div(r.rolling(20,min_periods=16).std().replace(0,np.nan))
sig=pd.DataFrame(index=c.index,columns=A,dtype=float)
for dt in c.index:
 q=pd.concat([raw.loc[dt],trend.loc[dt]],axis=1).dropna()
 if len(q)>=8:
  x=q.iloc[:,1].values; y=q.iloc[:,0].values
  b=np.cov(x,y,ddof=1)[0,1]/np.var(x,ddof=1) if np.var(x,ddof=1)>0 else 0.
  sig.loc[dt,q.index]=y-b*(x-x.mean())
print('candidate=trend_orthogonal_residual_drawdown_containment_20_60 cutoff=',c.dropna(how='all').index.max().date())
print('signal cells',int(sig.notna().sum().sum()),'of',sig.size,'coverage',round(sig.notna().sum().sum()/sig.size,5))
for h in [1,5,10,20]:
 f=c.shift(-h)/c-1; vals=[]; ns=[]; ds=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z): vals.append(z); ns.append(len(q)); ds.append(dt)
 v=np.array(vals); print('h',h,'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),4),'dates',len(v),'mean_n',round(np.mean(ns),2),'min_n',min(ns))
 for lab,lo,hi in [('2020-2027','2020-01-01','2027-12-31'),('2028-2030','2028-01-01','2030-12-31'),('2031-current','2031-01-01','2034-11-22'),('latest_6m','2034-05-22','2034-11-22')]:
  x=np.array([z for z,d in zip(v,ds) if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)])
  if len(x)>1: print(' ',lab,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
t=[]
for i in range(1,len(sig)):
 q=pd.concat([sig.iloc[i-1],sig.iloc[i]],axis=1).dropna()
 if len(q)>=8:t.append(np.abs(q.iloc[:,0].rank(pct=True)-q.iloc[:,1].rank(pct=True)).mean())
print('rank_turnover',round(np.mean(t),6),'adjacent_dates',len(t),'median_iqr',round(sig.quantile(.75,axis=1).sub(sig.quantile(.25,axis=1)).median(),8))
sig.to_pickle('scripts/miner_1_20341123_trend_orthogonal_residual_drawdown_containment_20_60_signal.pkl')

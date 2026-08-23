import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:return pd.Series(dtype=float)
 return d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close.astype(float).sort_index()
P=pd.DataFrame({s:load(s) for s in U}).sort_index().loc[:pd.Timestamp('2030-10-30')]
R=P.pct_change(); market=R.mean(axis=1)
rm=P/P.shift(20)-1; mret=(1+market).rolling(20).apply(np.prod,raw=True)-1
# Compute each asset's causal beta explicitly to avoid pandas covariance alignment differences.
beta=pd.DataFrame(index=P.index,columns=U,dtype=float)
for s in U:
 beta[s]=R[s].rolling(60,min_periods=40).cov(market)/market.rolling(60,min_periods=40).var()
res=rm-beta.mul(mret,axis=0); rres=R-beta.mul(market,axis=0)
vol=rres.rolling(60,min_periods=40).std()*np.sqrt(252)
sig=(-res/vol).replace([np.inf,-np.inf],np.nan); fwd=P.shift(-10)/P-1
obs=[];prev=None;turns=[];art=[]
for dt in sig.index:
 x=sig.loc[dt].dropna();y=fwd.loc[dt].reindex(x.index).dropna();x=x.reindex(y.index)
 if len(x)<8:continue
 obs.append((dt,x.corr(y,method='spearman'),len(x)));q=x.rank(pct=True)
 if prev is not None:turns.append((q-prev.reindex(q.index)).abs().mean())
 prev=q;art += [{'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)} for s,v in x.items()]
z=pd.DataFrame(obs,columns=['date','ic','n']).dropna();m=z.ic.mean();sd=z.ic.std(ddof=1)
print('candidate residualized_cross_asset_reversal_20d');print('assets',15,'dates',len(z),'meanN',z.n.mean(),'coverage',z.n.mean()/15,'IC',m,'daily_ICIR',m/sd,'hit',(z.ic>0).mean(),'turnover',np.mean(turns))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-10-30')]:
 q=z[(z.date>=a)&(z.date<=b)];print(a,b,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,20]:
 q=[]
 for dt in sig.index:
  x=sig.loc[dt].dropna();y=(P.shift(-h)/P-1).loc[dt].reindex(x.index).dropna();x=x.reindex(y.index)
  if len(x)>=8:q.append(x.corr(y,method='spearman'))
 print('decay',h,'dates',len(q),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1))
pd.DataFrame(art).to_csv('scripts/miner_2_20301031_residualized_cross_asset_reversal_signal.csv',index=False)

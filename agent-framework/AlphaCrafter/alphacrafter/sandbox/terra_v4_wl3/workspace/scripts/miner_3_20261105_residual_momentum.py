import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').close
 P[s]=d
P=pd.DataFrame(P).sort_index().loc[:'2026-11-04']; R=P.pct_change(fill_method=None)
# candidate: beta-neutralized medium-term momentum; beta estimated trailing 60d to equal-weight cross-asset market
m=R.mean(axis=1); cov=R.rolling(60,min_periods=40).cov(m); var=m.rolling(60,min_periods=40).var()
beta=cov.div(var,axis=0)
raw=R.rolling(20,min_periods=20).sum(); market=m.rolling(20,min_periods=20).sum()
F=raw-beta.mul(market,axis=0)
# rank cross section, forward returns
for h in [1,5,10]:
 ic=[]; ns=[]
 for i in range(len(P)-h):
  f=F.iloc[i]; y=P.pct_change(h,fill_method=None).shift(-h).iloc[i]
  z=pd.concat([f,y],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(ic); print(h,'dates',len(a),'avgN',np.mean(ns),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
# coverage turnover
valid=F.notna().sum(axis=1); print('coverage',valid.sum()/(len(F)*15),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean(),'dates',len(P))
for y in range(2020,2027):
 a=[]
 for i in range(len(P)-1):
  if P.index[i].year==y:
   z=pd.concat([F.iloc[i],P.pct_change().shift(-1).iloc[i]],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(y,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1) if len(a)>1 else np.nan)
F.to_csv('scripts/miner_3_20261105_residual_momentum_signal.csv',index_label='date')

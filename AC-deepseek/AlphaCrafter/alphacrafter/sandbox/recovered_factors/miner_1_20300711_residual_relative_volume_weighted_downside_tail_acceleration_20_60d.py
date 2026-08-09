"""Validate one idea: continuous relative-volume weighted residual downside-tail acceleration."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-07-10')
def load(a,col):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 return d.loc[:END,col].astype(float)
p=pd.DataFrame({a:load(a,'close') for a in A}); v=pd.DataFrame({a:load(a,'volume') for a in A})
r=p.pct_change(); market=r.mean(axis=1)
beta=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(market)/(market.rolling(60,min_periods=40).var()+1e-12) for a in A})
e=r-beta.mul(market,axis=0)
# Continuous own-relative-volume weighting avoids sparse event selection.  Cap weights
# to prevent a single anomalous print dominating; compare 20d intensity with 60d baseline.
rv=(v/(v.rolling(20,min_periods=12).median()+1e-12)).clip(lower=0,upper=3)
tail=(e.clip(upper=0)**2)*rv
short=tail.rolling(20,min_periods=12).mean(); long=tail.rolling(60,min_periods=35).mean()
f=short/(long+1e-12)-1
print('FACTOR residual_relative_volume_weighted_downside_tail_acceleration_20_60d','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'coverage',round(float(f.notna().mean().mean()),6),'mean_relative_volume',round(float(rv.mean().mean()),6))
ics={}; metrics={}
for h in [1,5,10,20]:
 out=[]; ns=[]; fw=p.shift(-h)/p-1
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   out.append((t,z.f.corr(z.y,method='spearman'))); ns.append(len(z))
 x=pd.Series(dict(out),dtype=float); ics[h]=x; sd=x.std(ddof=1); metrics[h]=(x.mean(),x.mean()/sd,(x>0).mean(),len(x),float(np.mean(ns)))
 print('HORIZON',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'HIT',round((x>0).mean(),6),'DATES',len(x),'MEAN_N',round(float(np.mean(ns)),3))
# report each horizon's broad early, mid and contemporary regimes
for h,x in ics.items():
 for n,mask in [('2020_24',x.index<pd.Timestamp('2025-01-01')),('2025_26',(x.index>=pd.Timestamp('2025-01-01'))&(x.index<pd.Timestamp('2027-01-01'))),('2027_onward',x.index>=pd.Timestamp('2027-01-01'))]:
  q=x[mask]; print('REGIME',h,n,'DATES',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'HIT',round((q>0).mean(),6))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('RANK_TURNOVER',round(float(np.mean(turn)),6),'TURNOVER_DATES',len(turn))
print('DECAY',json.dumps({str(h):{'ic':round(float(q[0]),6),'icir':round(float(q[1]),6),'dates':q[3]} for h,q in metrics.items()}))
print('LIBRARY_CORRELATION_NOT_COMPUTED unless a same-horizon IC and ICIR gate passes.')

"""Miner 1: validate residual return-distribution asymmetry shift (one factor idea)."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-06-12')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float)
p=pd.DataFrame({a:load(a) for a in A}); r=p.pct_change(); market=r.mean(axis=1)
beta=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(market)/(market.rolling(60,min_periods=40).var()+1e-12) for a in A})
e=r-beta.mul(market,axis=0)
# Difference between current and structural residual return skewness: a distribution-shape, not macro loading feature.
def skew(x):
 s=x.std(ddof=0)
 return ((x-x.mean())**3).mean()/(s**3+1e-12)
sk20=e.rolling(20,min_periods=15).apply(skew,raw=False); sk60=e.rolling(60,min_periods=45).apply(skew,raw=False)
f=sk20-sk60
print('FACTOR residual_skewness_shift_20_60d','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'coverage',round(float(f.notna().mean().mean()),6))
ics={}; met={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 met[h]=(x.mean(),x.mean()/sd,(x>0).mean(),len(x),np.mean(ns))
 print('HORIZON',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'HIT',round((x>0).mean(),6),'DATES',len(x),'MEAN_N',round(float(np.mean(ns)),3))
for label,mask in [('2020_24',ics[10].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 x=ics[10][mask]; print('REGIME10',label,'DATES',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'HIT',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); tr=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: tr.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('RANK_TURNOVER',round(float(np.mean(tr)),6),'TURNOVER_DATES',len(tr))
print('DECAY',json.dumps({str(h):{'ic':round(float(v[0]),6),'icir':round(float(v[1]),6),'dates':int(v[3])} for h,v in met.items()}))
print('LIBRARY_CORRELATION_NOT_COMPUTED: computed only if predictive gate passes; missing evidence means no admission.')

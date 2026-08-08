"""Validation: 40-observation own-tail gain/loss asymmetry, standardized by total tail magnitude."""
import pandas as pd,numpy as np,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-01')
def rd(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
p=pd.DataFrame({a:rd(a).close.astype(float) for a in A}); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std(); peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
def tail_asym(x):
 def one(z):
  q1=np.quantile(z,.2);q4=np.quantile(z,.8); up=z[z>=q4].mean(); dn=z[z<=q1].mean()
  return (up+dn)/(up-dn) if up-dn>1e-12 else np.nan
 return x.rolling(40,min_periods=30).apply(one,raw=True)
f=r.apply(tail_asym)
print('FACTOR own_tail_gain_loss_asymmetry_40obs = (mean top-quintile daily return + mean bottom-quintile daily return)/(mean top-quintile return - mean bottom-quintile return), rolling 40 observations; higher means upside tails dominate downside tails')
print('visible_through',END.date(),'data_range',p.index.min().date(),p.index.max().date(),'assets',len(A))
def met(h):
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out)); sd=x.std();reg={}
 for name,mask in {'2020':x.index.year==2020,'2021_2022':x.index.year.isin([2021,2022]),'2023_2024':x.index.year.isin([2023,2024]),'2025_2026':x.index.year.isin([2025,2026]),'2027':x.index.year==2027}.items():
  q=x[mask];reg[name]={'dates':len(q),'ic':round(q.mean(),6),'icir':round(q.mean()/q.std(),6),'hit':round((q>0).mean(),4)}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'ic_dates':len(x),'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':reg}
print('coverage',int(f.count().sum()),'/',f.size,round(f.count().sum()/f.size,4))
for h in [1,5,10,20]:print('METRIC',json.dumps(met(h)))
# library signals (reconstructed identically to prior validation definitions)
def cbeta(x,y,cond,sign=1):
 x=x.where(cond);y=y.where(cond);return sign*x.rolling(40,min_periods=12).cov(y)/y.rolling(40,min_periods=12).var()
trend=(p/p.shift(20)-1)/vol; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol
orth=acc* np.nan
for d in r.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]]; orth.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
spxb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(r.SPX)/r.SPX.rolling(20,min_periods=15).var() for a in A})
kurt=-r.rolling(40,min_periods=30).kurt(); es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda z:np.mean(z[z<=np.quantile(z,.2)]),raw=True)/vol[a] for a in A})
low=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A});high=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A});asym=pd.DataFrame({a:cbeta(r[a],peer[a],peer[a]<0)-cbeta(r[a],peer[a],peer[a]>0) for a in A})
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(r.index);du=pd.DataFrame({a:cbeta(r[a],dxy,dxy>0,-1) for a in A});dd=pd.DataFrame({a:cbeta(r[a],dxy,dxy<0) for a in A})
part=pd.DataFrame({a:np.log(rd(a).volume.astype(float)/rd(a).volume.astype(float).rolling(20,min_periods=1).mean()) for a in A}); auto=r.rolling(20,min_periods=15).corr(r.shift(1))
lib={'risk_adjusted_trend':trend,'volnorm_reversal':rev,'relative_volume':part,'orthogonal_acceleration':orth,'negative_spx_beta':spxb,'inverse_kurtosis':kurt,'inverse_expected_shortfall':es,'downside_peer_correlation':low,'inverse_upside_peer_correlation':high,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'return_autocorrelation':auto,'asymmetric_peer_beta_resilience':asym}
mx=-1
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();q=z.f.corr(z.x,method='spearman');print('LIB',n,round(q,6),len(z));
 if abs(q)>mx:mx=abs(q);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who)

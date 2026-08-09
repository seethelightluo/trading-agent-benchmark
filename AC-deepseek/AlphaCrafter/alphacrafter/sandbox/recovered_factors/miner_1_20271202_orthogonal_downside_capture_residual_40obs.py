"""One-factor validation: cross-sectionally orthogonalized downside-capture residual."""
import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2027-12-01')
def rd(a):
 d=get_stock_daily_data(a,5000).set_index('date'); d.index=pd.to_datetime(d.index)
 return d.loc[:END]
p=pd.DataFrame({a:pd.to_numeric(rd(a).close,errors='coerce') for a in A}); r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
peer=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
# Candidate: 40-observation mean own return on own leave-one-out peer-down days / 20d volatility,
# residualized each day on the two admitted conditional-correlation signals.
raw=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]<0)) for a in A})
invup=pd.DataFrame({a:-r[a].where(peer[a]>0).rolling(40,min_periods=12).corr(peer[a].where(peer[a]>0)) for a in A})
f=raw*np.nan
for d in p.index:
 z=pd.concat([raw.loc[d].rename('y'),down.loc[d].rename('d'),invup.loc[d].rename('u')],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z[['d','u']]]; f.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
def met(h):
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(); regs={}
 for n,yrs in {'2020':[2020],'2021_2022':[2021,2022],'2023_2024':[2023,2024],'2025_2026':[2025,2026],'2027':[2027]}.items():
  y=x[x.index.year.isin(yrs)]; regs[n]={'dates':len(y),'ic':float(y.mean()),'icir':float(y.mean()/y.std()) if len(y)>1 and y.std()>0 else None,'hit':float((y>0).mean()) if len(y) else None}
 turn=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(sd/np.sqrt(len(x))),'mean_instruments':float(np.mean(ns)),'turnover_10d':float(np.mean(turn)),'regimes':regs}
print('FACTOR orthogonal_downside_capture_residual_40obs; visible_through',END.date(),'assets',len(A),'data_range',p.index.min().date(),p.index.max().date())
print('coverage',int(f.count().sum()),'/',f.size,float(f.count().sum()/f.size))
for h in [1,5,10,20]:print('METRIC',json.dumps(met(h)))
# Exact admitted factors/reconstructions required for library gate.
trend=(p/p.shift(20)-1)/vol; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol
orth=acc*np.nan
for d in p.index:
 z=pd.concat([acc.loc[d],trend.loc[d]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]];orth.loc[d,z.index]=z.iloc[:,0]-X@np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
spxb=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(r.SPX)/r.SPX.rolling(20,min_periods=15).var() for a in A}); kurt=-r.rolling(40,min_periods=30).kurt(); es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x:np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A}); autoc=r.rolling(20,min_periods=15).corr(r.shift(1))
dxy=get_index_daily_data('DXY',5000).set_index('date');dxy.index=pd.to_datetime(dxy.index);dx=pd.to_numeric(dxy.loc[:END,'close'],errors='coerce').pct_change().reindex(r.index)
def beta(x,y,c,sgn=1):
 return sgn*x.where(c).rolling(40,min_periods=12).cov(y.where(c))/y.where(c).rolling(40,min_periods=12).var()
du=pd.DataFrame({a:beta(r[a],dx,dx>0,-1) for a in A});dd=pd.DataFrame({a:beta(r[a],dx,dx<0) for a in A});asym=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0)-beta(r[a],peer[a],peer[a]>0) for a in A})
part=pd.DataFrame({a:np.log(pd.to_numeric(rd(a).volume,errors='coerce')/pd.to_numeric(rd(a).volume,errors='coerce').rolling(20,min_periods=1).mean()) for a in A})
lib={'risk_adjusted_trend':trend,'volnorm_reversal':rev,'relative_volume':part,'orthogonal_acceleration':orth,'negative_spx_beta':spxb,'inverse_kurtosis':kurt,'inverse_expected_shortfall':es,'downside_peer_correlation':down,'inverse_upside_peer_correlation':invup,'negative_conditional_dxy_up_beta':du,'positive_conditional_dxy_down_beta':dd,'return_autocorrelation':autoc,'asymmetric_peer_beta_resilience':asym}
mx=0
for n,x in lib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); rho=z.f.corr(z.x,method='spearman');print('LIB',n,round(rho,6),len(z));
 if abs(rho)>mx:mx=abs(rho);who=n
print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who)

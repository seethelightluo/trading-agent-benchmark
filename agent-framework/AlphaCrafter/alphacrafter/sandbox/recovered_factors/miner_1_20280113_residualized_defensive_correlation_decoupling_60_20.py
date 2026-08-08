"""miner_1: one idea -- residualized defensive-correlation decoupling."""
import pathlib,json,numpy as np,pandas as pd
# Re-use the point-in-time loading / residualization and full reconstructed library.
src=pathlib.Path('scripts/miner_1_20271230_residualized_market_correlation_dispersion_shift_60_20.py').read_text()
src=src.replace("END=pd.Timestamp('2027-12-29')", "END=pd.Timestamp('2028-01-12')")
# Only execute loader helpers, prior to candidate construction.
exec(src.split('# IDEA:')[0])
# IDEA: rising decoupling from the defensive basket may identify assets with
# idiosyncratic return drivers. Remove short trend and own risk cross-sectionally.
defensive=r[['XAU','US10Y','CN10Y']].mean(axis=1)
c20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(defensive) for a in A})
c60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(defensive) for a in A})
raw=-(c20-c60)
trend=(p/p.shift(20)-1)/own
f=residual(raw,trend,own)
# Same library reconstruction used by prior validation, including later proxies.
start=src.index("start=src.index(\"lib={'miner_3_risk_adjusted_trend_20d'\")")
# Extract executed library block directly from old source by delimiters.
oldstart=src.index("lib={'miner_3_risk_adjusted_trend_20d'")
oldend=src.index("print('FACTOR",oldstart)
exec(src[oldstart:oldend])
acraw=r.rolling(20,min_periods=15).apply(lambda x:x.dropna().autocorr(lag=1) if len(x.dropna())>=15 else np.nan,raw=False)
lib['miner_1_residualized_return_autocorrelation_20d']=residual(acraw,trend,own)
signraw=np.sign(r).rolling(20,min_periods=15).mean()
lib['miner_1_residualized_sign_persistence_proxy_20d']=residual(signraw,trend,own)
print('FACTOR residualized_defensive_correlation_decoupling_60_20 cutoff',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_proxies',len(lib))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals)); sd=x.std(ddof=1)
 q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}; metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==10:
  for name,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_28',x.index>='2025')]:
   y=x[mask]; print('REGIME',name,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.mean(tos),6),'TURNOVER_DATES',len(tos))
mx=-1;winner=None;wcells=0
for name,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman'); print('LIBRARY',name,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);winner=name;wcells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'CELLS',wcells,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))

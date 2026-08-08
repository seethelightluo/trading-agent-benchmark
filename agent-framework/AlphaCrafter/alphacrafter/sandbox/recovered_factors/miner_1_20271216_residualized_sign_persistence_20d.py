"""miner_1: validate one idea -- residualized sign-persistence imbalance."""
import pathlib,json,numpy as np,pandas as pd
# Reuse only the point-in-time loader/helper setup from a prior research script.
src=pathlib.Path('scripts/miner_1_20271118_residualized_return_autocorrelation_20d.py').read_text()
src=src.replace("END=pd.Timestamp('2027-11-17')", "END=pd.Timestamp('2027-12-15')")
exec(src.split('# Lag-one autocorrelation')[0])
# Candidate: fraction of positive daily returns minus fraction negative over 20 sessions.
# Residualize against return level (risk-adjusted trend) and own risk: captures path
# consistency rather than simply a large cumulative move.
raw=np.sign(r).rolling(20,min_periods=15).mean()
trend=(p/p.shift(20)-1)/own
f=residual(raw,trend,own)
# Recreate all active factor proxies, starting from prior complete library construction.
old=pathlib.Path('scripts/miner_1_20271118_residualized_return_autocorrelation_20d.py').read_text()
block=old.split("# Complete reconstructed admitted-factor signal set")[1].split("print('FACTOR")[0]
# avoid its old candidate/raw definition and execute its library definitions in this context
exec(block)
# Factors admitted after the 2027-11-17 reference script.
acraw=r.rolling(20,min_periods=15).apply(lambda x:x.dropna().autocorr(lag=1) if len(x.dropna())>=15 else np.nan,raw=False)
lib['miner_1_residualized_return_autocorrelation_20d']=residual(acraw,trend,own)
# residual dispersion shock resilience: lower latest residual dispersion relative to its baseline.
disp=e.rolling(20,min_periods=15).std()/e.rolling(60,min_periods=40).std()
lib['miner_3_residual_dispersion_shock_resilience_60d']=-disp
lib['miner_3_residual_upside_volume_confirmation_60d']=np.maximum(p/p.shift(20)-1,0)*vol.tail(1).reindex(p.index,method='ffill')/vol.rolling(60,min_periods=40).mean()
print('FACTOR residualized_sign_persistence_20d cutoff',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_proxies',len(lib))
metrics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals));sd=x.std(ddof=1);q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
 if h==10:
  for name,mask in [('2020',(x.index<'2021')),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025_27',x.index>='2025')]:
   y=x[mask];print('REGIME',name,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(np.mean(tos),6),'TURNOVER_DATES',len(tos))
mx=-1;winner=None;wcells=0
for name,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');print('LIBRARY',name,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);winner=name;wcells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'CELLS',wcells,'DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))

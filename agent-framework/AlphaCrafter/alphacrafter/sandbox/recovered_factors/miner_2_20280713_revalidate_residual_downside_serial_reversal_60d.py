"""Miner 2 scheduled revalidation: Residual Downside Serial Reversal (60d), through 2028-07-12."""
import pathlib,json,numpy as np,pandas as pd
# Reuse the complete current-library reconstruction; its END literal is updated before execution.
src=pathlib.Path('scripts/miner_3_20280504_residual_downside_volume_deceleration_complete_library.py').read_text()
src=src.replace("END=pd.Timestamp('2028-05-03')","END=pd.Timestamp('2028-07-12')")
exec(src)
# Documented candidate: negative lag-one autocorrelation of downside-only 60d beta-neutral residual.
down=e.clip(upper=0)
def ac(x):
 z=pd.DataFrame({'x':x,'lag':x.shift(1)}).dropna()
 return z.x.corr(z.lag) if len(z)>=45 else np.nan
f=-down.rolling(60,min_periods=45).apply(ac,raw=False)
print('REVALIDATION miner_2_residual_downside_serial_reversal_60d END',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
metrics={}; IC={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: out.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out));IC[h]=x;sd=x.std(ddof=1)
 q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
x=IC[20]
for n,m in [('2025_26',(x.index>='2025')&(x.index<'2027')),('2027_28',(x.index>='2027'))]:
 y=x[m];print('REGIME20',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),6),'names',round(float(np.mean([len(pd.concat([f.loc[t],(p.shift(-20)/p-1).loc[t]],axis=1).dropna()) for t in y.index])),4))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to),'LATEST_VALID',int(f.iloc[-1].notna().sum()))
mx=-1;winner=None;cells=0
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman')
 print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
 if abs(rho)>mx:mx=abs(rho);winner=n;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'CELLS',cells)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'hit':round(q['ic_hit_ratio'],6),'dates':q['ic_dates'],'names':round(q['mean_valid_instruments'],4)} for h,q in metrics.items()}))

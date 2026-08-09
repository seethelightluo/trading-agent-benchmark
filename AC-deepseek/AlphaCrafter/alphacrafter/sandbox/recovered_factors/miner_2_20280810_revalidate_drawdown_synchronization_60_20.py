"""Scheduled complete-library revalidation of Drawdown-Synchronization Improvement, visible through 2028-08-09."""
import pathlib,json,numpy as np,pandas as pd,io,contextlib
src=pathlib.Path('scripts/miner_3_20280504_residual_downside_volume_deceleration_complete_library.py').read_text().replace("END=pd.Timestamp('2028-05-03')","END=pd.Timestamp('2028-08-09')")
with contextlib.redirect_stdout(io.StringIO()): exec(src)
# Improvement in correlation of returns with change in cross-asset drawdown breadth, 60d prior vs 20d current.
dd=p/p.rolling(60,min_periods=40).max()-1
breadth=(dd<-.05).mean(axis=1)
sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A})
f=sy.shift(20)-sy
lib['miner_2_drawdown_synchronization_improvement_60_20']=f
print('REVALIDATION miner_2_drawdown_synchronization_improvement_60_20 END',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib)-1)
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((t,z.f.corr(z.y,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals)); ics[h]=x; sd=x.std(ddof=1)
 q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}; metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
x=ics[20]
for name,mask in [('2025_2026',(x.index>='2025')&(x.index<'2027')),('2027_2028',(x.index>='2027'))]:
 y=x[mask];print('REGIME20',name,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),6))
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(tos)),6),'TURNOVER_DATES',len(tos),'LATEST_VALID',int(f.iloc[-1].notna().sum()))
mx=-1;winner=None;cells=0
for n,s in sorted(lib.items()):
 if n=='miner_2_drawdown_synchronization_improvement_60_20':continue
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman')
 if abs(rho)>mx:mx=abs(rho);winner=n;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'CELLS',cells)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'hit':round(q['ic_hit_ratio'],6),'dates':q['ic_dates'],'names':round(q['mean_valid_instruments'],4)} for h,q in metrics.items()}))

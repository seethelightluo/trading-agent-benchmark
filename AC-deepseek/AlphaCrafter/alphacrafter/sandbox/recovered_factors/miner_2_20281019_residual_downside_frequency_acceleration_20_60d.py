"""One candidate: residual downside-frequency acceleration, 20d versus 60d.
Tests whether a recent rise in the frequency (not magnitude) of idiosyncratic down days
predicts cross-asset forward returns. Visibility-safe through 2028-10-18.
"""
import pathlib,json,numpy as np,pandas as pd,io,contextlib
# Reuse visible-data loader and exhaustive historical library construction, without its candidate.
src=pathlib.Path('scripts/miner_3_20280907_residual_breadth_shock_sensitivity_contraction_20_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2028-09-06')","END=pd.Timestamp('2028-10-18')")
with contextlib.redirect_stdout(io.StringIO()): exec(src.split('# Breadth shock')[0])
# Add post-May admitted signals not present in inherited scaffold.
# Residual downside signed volume pressure deceleration.
dsv=(-e).clip(lower=0)*vs
lib['miner_3_residual_downside_signed_volume_pressure_deceleration_20_60d']=-(dsv.rolling(20,min_periods=12).mean()/(e.rolling(20,min_periods=15).std()+1e-12)-dsv.rolling(60,min_periods=25).mean()/(e.rolling(60,min_periods=40).std()+1e-12))
# Drawdown-weighted relative participation acceleration.
dd=1-p/p.rolling(10,min_periods=8).max(); q=(vs.rank(axis=1,pct=True)-.5)*dd
lib['miner_3_drawdown_weighted_relative_participation_rank_acceleration_20_60d']=q.rolling(20,min_periods=12).mean()-q.rolling(60,min_periods=35).mean()
# Residual breadth- and return-dispersion-shock beta expansions.
breadth=(r>0).mean(axis=1).diff(); disp=e.std(axis=1,ddof=0).diff()
for key,shock in [('miner_3_residual_breadth_shock_sensitivity_expansion_20_60d',breadth),('miner_3_residual_return_dispersion_shock_sensitivity_expansion_20_60d',disp)]:
 b20=pd.DataFrame({a:e[a].rolling(20,min_periods=14).cov(shock)/(shock.rolling(20,min_periods=14).var()+1e-12) for a in A})
 b60=pd.DataFrame({a:e[a].rolling(60,min_periods=42).cov(shock)/(shock.rolling(60,min_periods=42).var()+1e-12) for a in A})
 lib[key]=b20-b60
# Candidate: downside incidence acceleration. Positive means an asset has had more
# frequent residual-loss sessions recently than over its own trailing baseline.
f=(e.lt(0).rolling(20,min_periods=14).mean()-e.lt(0).rolling(60,min_periods=42).mean())
print('CANDIDATE residual_downside_frequency_acceleration_20_60d END',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'other_library',len(lib))
metrics={}; IC={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p)-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   rho=z.f.corr(z.y,method='spearman')
   if pd.notna(rho): out.append((t,rho));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);IC[h]=x; sd=x.std(ddof=1)
 q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
x=IC[20]
for n,m in [('2025_2026',(x.index>='2025')&(x.index<'2027')),('2027_2028',(x.index>='2027'))]:
 y=x[m]; print('REGIME20',n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),6))
rk=f.rank(axis=1,pct=True);tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(tos)),6),'TURNOVER_DATES',len(tos),'LATEST_VALID',int(f.iloc[-1].notna().sum()))
mx=-1;winner=None;cells=0
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman')
 print('SCREEN',n,'rho',round(rho,6),'cells',len(z))
 if len(z) and abs(rho)>mx:mx=abs(rho);winner=n;cells=len(z)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'CELLS',cells)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'hit':round(q['ic_hit_ratio'],6),'dates':q['ic_dates'],'names':round(q['mean_valid_instruments'],4)} for h,q in metrics.items()}))

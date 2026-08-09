"""Miner 1: downside-only USD-shock beta transition (60/20), cutoff 2028-12-27.
Higher values indicate declining beta to completed-day DXY selloffs (negative DXY returns), residualized versus 20d trend and own volatility.
"""
import pathlib,json,numpy as np,pandas as pd,io,contextlib
src=pathlib.Path('scripts/miner_3_20280504_residual_downside_volume_deceleration_complete_library.py').read_text().replace("END=pd.Timestamp('2028-05-03')","END=pd.Timestamp('2028-12-27')")
with contextlib.redirect_stdout(io.StringIO()): exec(src)
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END,'close'].astype(float).reindex(p.index).ffill()
dr=dxy.pct_change(); shock=(-dr.where(dr<0,0.0))
def beta(w,mp):
 return pd.DataFrame({a:r[a].rolling(w,min_periods=mp).cov(shock)/shock.rolling(w,min_periods=mp).var() for a in A})
raw=-(beta(20,12)-beta(60,40))
trend=(p/p.shift(20)-1)/own
f=residual(raw,trend,own)
# Add the closely related admitted absolute-DXY transition comparator explicitly.
abs_shock=dr.abs()
def abeta(w,mp):
 return pd.DataFrame({a:r[a].rolling(w,min_periods=mp).cov(abs_shock)/abs_shock.rolling(w,min_periods=mp).var() for a in A})
absfactor=residual(-(abeta(20,12)-abeta(60,40)),trend,own)
lib['miner_1_residualized_usd_shock_beta_transition_60_20']=absfactor
print('FACTOR residualized_downside_usd_shock_beta_transition_60_20 validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_proxies',len(lib))
metrics={}; IC={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p)-1; out=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): out.append((t,q)); ns.append(len(z))
 x=pd.Series(dict(out),dtype=float); IC[h]=x; sd=x.std(ddof=1)
 q={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)};metrics[h]=q
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in q.items()}))
for h in [10,20]:
 x=IC[h]
 for n,m in [('2020_21',x.index<'2022'),('2022_23',(x.index>='2022')&(x.index<'2024')),('2024_25',(x.index>='2024')&(x.index<'2026')),('2026_28',x.index>='2026')]:
  y=x[m]; print('REGIME',h,n,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6) if len(y)>1 else None,'hit',round((y>0).mean(),4))
rk=f.rank(axis=1,pct=True); to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
mx=-1;winner=None;cells=0
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman')
 if len(z) and abs(rho)>mx:mx=abs(rho);winner=n;cells=len(z)
 print('LIBRARY',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',winner,'CELLS',cells)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'hit':round(q['ic_hit_ratio'],6),'dates':q['ic_dates']} for h,q in metrics.items()}))

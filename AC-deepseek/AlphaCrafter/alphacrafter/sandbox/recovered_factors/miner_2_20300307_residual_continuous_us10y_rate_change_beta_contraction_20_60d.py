"""Validate one idea: residual continuous US10Y rate-change beta contraction (20d vs 60d)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2030-03-06')").replace("No inherited execution; all series are restricted to 2029-02-21 (latest completed day before the decision date).","")
exec(prefix,globals())
# Smooth every-session rate-change exposure, standardized solely using a trailing 60d history.
dy=p['US10Y'].ffill().diff(); rate_change=dy/dy.rolling(60,min_periods=40).std().replace(0,np.nan)
def beta(a,w,n): return e[a].rolling(w,min_periods=n).cov(rate_change)/rate_change.rolling(w,min_periods=n).var()
f=pd.DataFrame({a:-(beta(a,20,14)-beta(a,60,42)) for a in A})
print('FACTOR residual_continuous_us10y_rate_change_beta_contraction_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'rate_change_coverage',round(float(rate_change.notna().mean()),6),flush=True)
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q):out.append((t,q));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}),flush=True)
if len(ics[10]):
 for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_28',(ics[10].index>=pd.Timestamp('2027-01-01'))&(ics[10].index<pd.Timestamp('2029-01-01'))),('2029_onward',ics[10].index>=pd.Timestamp('2029-01-01'))]:
  x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turns=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(z)))
if screen:
 mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))

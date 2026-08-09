"""One candidate: leave-one-out cross-asset market beta contraction (60d/20d).
For each asset use the contemporaneous median daily return of the other 14 tradable
assets as an observable broad-market driver; factor is 60d beta minus 20d beta.
It tests whether a recently falling common-risk loading predicts relative returns."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20310123_positive_usdjpy_change_shock_loading_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# Candidate: sensitivity to directly observable *positive* yen')[0].replace("END=pd.Timestamp('2031-01-22')","END=pd.Timestamp('2032-10-13')")
exec(prefix,globals())
rr=p.pct_change()
# Leave-one-out median avoids an asset mechanically entering its own driver.
f=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
 d=rr.drop(columns=a).median(axis=1)
 def beta(w,minp):
  return rr[a].rolling(w,min_periods=minp).cov(d)/d.rolling(w,min_periods=minp).var()
 f[a]=beta(60,42)-beta(20,14)
print('FACTOR leave_one_out_cross_asset_market_beta_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'driver_coverage',round(float(rr.notna().mean().mean()),6),'latest250_unique_minmax',int(f.tail(250).nunique(axis=1).min()),int(f.tail(250).nunique(axis=1).max()))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v):out.append((t,v));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[20].index<pd.Timestamp('2025')),('2025_26',(ics[20].index>=pd.Timestamp('2025'))&(ics[20].index<pd.Timestamp('2027'))),('2027_onward',ics[20].index>=pd.Timestamp('2027'))]:
 x=ics[20][mask];print('REGIME20',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None,'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn),'VALID_CELLS',int(f.notna().sum().sum()))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(q)))
if screen:
 mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION EVIDENCE_MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))

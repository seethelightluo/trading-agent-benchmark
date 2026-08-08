"""One candidate: signed linear residual tail-severity and downside-recovery deterioration.
Tests the additive (rather than interaction) form: assets with recently worsening
idiosyncratic tail severity and worsening post-shock residual recovery score high.
Components are cross-sectional ranks, making their units comparable."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20311127_residual_downside_tail_severity_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# For each window')[0].replace("END=pd.Timestamp('2031-11-26')","END=pd.Timestamp('2033-03-02')")
exec(prefix,globals())
def tail_severity(x,w,n):
 def calc(z):
  z=np.asarray(z,float); s=np.std(z,ddof=1); tail=z[z < -s]
  return -tail.mean()/(s+1e-12) if len(tail) else 0.0
 return x.rolling(w,min_periods=n).apply(calc,raw=True)
def downside_recovery(x,w,n):
 def calc(z):
  z=np.asarray(z,float); s=np.std(z[:-1],ddof=1)
  if not np.isfinite(s) or s<1e-12:return np.nan
  shock=z[:-1] < -s
  return np.mean(z[1:][shock])/s if shock.any() else 0.0
 return x.rolling(w,min_periods=n).apply(calc,raw=True)
sev=pd.DataFrame({a:tail_severity(e[a],20,14)-tail_severity(e[a],60,42) for a in A})
# Negative recovery change is deterioration: following residual shocks has become weaker.
rec=-pd.DataFrame({a:downside_recovery(e[a],20,14)-downside_recovery(e[a],60,42) for a in A})
f=(sev.rank(axis=1,pct=True)-.5)+(rec.rank(axis=1,pct=True)-.5)
print('FACTOR signed_linear_residual_tail_severity_inverse_downside_recovery_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
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
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None,'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn),'VALID_CELLS',int(f.notna().sum().sum()))
screen=[]
for name,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),name,rho,len(q)))
if screen:
 mx,name,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',name,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION EVIDENCE_MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))

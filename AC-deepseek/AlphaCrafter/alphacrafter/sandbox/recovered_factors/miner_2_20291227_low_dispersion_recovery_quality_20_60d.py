"""Miner 2: validate one idea, low-dispersion conditional residual recovery quality (20d vs 60d)."""
import json, numpy as np, pandas as pd, builtins, re
# Force every nested inherited loader to honor the current point-in-time cutoff.
END_TARGET='2029-12-26'
_real_open=builtins.open
def patched_open(path,*args,**kwargs):
    h=_real_open(path,*args,**kwargs)
    if isinstance(path,str) and path.startswith('scripts/') and 'r' in (args[0] if args else kwargs.get('mode','r')):
        text=h.read(); h.close()
        text=re.sub(r"END=pd.Timestamp\('\d{4}-\d{2}-\d{2}'\)", "END=pd.Timestamp('"+END_TARGET+"')", text)
        import io
        return io.StringIO(text)
    return h
builtins.open=patched_open
src=_real_open('scripts/miner_3_20291213_low_dispersion_conditional_recovery_quality_20_60d.py',encoding='utf8').read()
prefix=src.split('# Candidate: quality')[0]
exec(prefix,globals())
builtins.open=_real_open
# One idea: after an own 10d drawdown, measure positive residual-return capture
# only while cross-asset residual dispersion is below its trailing median.  The
# 20d-minus-60d ratio captures acceleration in orderly recovery quality.
own_dd=(1-p.shift(1)/p.shift(1).rolling(10,min_periods=8).max()).clip(lower=0)
disp=e.std(axis=1,ddof=0); compressed=disp < disp.rolling(60,min_periods=40).median()
quality=e.clip(lower=0).mul(own_dd).where(compressed,0.0)
def score(w,n): return quality.rolling(w,min_periods=n).mean()/(own_dd.where(compressed,0.0).rolling(w,min_periods=n).mean()+1e-12)
f=score(20,14)-score(60,42)
print('FACTOR miner_2_low_dispersion_conditional_residual_recovery_quality_acceleration_20_60d','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_reconstructed',len(lib))
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): out.append((t,q)); ns.append(len(z))
 x=pd.Series(dict(out)); ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<'2025'),('2025_26',(ics[10].index>='2025')&(ics[10].index<'2027')),('2027_onward',ics[10].index>='2027')]:
 x=ics[10][mask]; print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
screen=[]
for name,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman'); screen.append((abs(rho),name,rho,len(z)))
mx,name,rho,cells=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',name,'rho',round(float(rho),6),'cells',cells)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))

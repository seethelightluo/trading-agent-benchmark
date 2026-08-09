"""Validate one idea: residual quantile-tail severity compression (20d vs 60d)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2029-11-14')")
exec(prefix,globals())
# Asset-local residual tail severity: mean distance below each rolling 20th
# percentile, normalized by the local residual scale. The score is structural
# (60d) minus recent (20d) severity: high values indicate tail compression.
def severity(x,w,n):
 q=x.rolling(w,min_periods=n).quantile(.20)
 excess=(q-x).clip(lower=0)
 return excess.rolling(w,min_periods=n).mean()/(x.rolling(w,min_periods=n).std()+1e-12)
f=severity(e,60,42)-severity(e,20,14)
fid='miner_1_residual_quantile_tail_severity_compression_20_60d'; lib.pop(fid,None)
print('FACTOR',fid,'VALIDATION_END',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'comparison_library',len(lib),'factor_cells',int(f.notna().sum().sum()))
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): out.append((t,q)); ns.append(len(z))
 x=pd.Series(dict(out),dtype=float); ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[5].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[5].index>=pd.Timestamp('2025-01-01'))&(ics[5].index<pd.Timestamp('2027-01-01'))),('2027_28',(ics[5].index>=pd.Timestamp('2027-01-01'))&(ics[5].index<pd.Timestamp('2029-01-01'))),('2029_onward',ics[5].index>=pd.Timestamp('2029-01-01'))]:
 x=ics[5][mask]; print('REGIME5',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None,'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman')
 if pd.notna(rho): screen.append((abs(rho),n,rho,len(z)))
if screen:
 mx,n,rho,c=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'hit':round(float(q['ic_hit_ratio']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))

"""One candidate: residual correlation-network centrality expansion (60d/20d).
Tests whether an asset becoming more correlated with the residual cross-asset network
has a subsequent relative-return implication. Uses only completed closes through 2033-10-12.
"""
import json,numpy as np,pandas as pd
src=open('scripts/miner_2_20330915_residual_medium_term_trend_deceleration_60_20d.py',encoding='utf8').read()
prefix=src.split("# e is residualized")[0].replace("END=pd.Timestamp('2033-09-14')","END=pd.Timestamp('2033-10-12')")
exec(prefix,globals())
# For each date and asset, centrality is its mean pairwise residual-return correlation.
# Factor is recent (20d) centrality minus baseline (60d); high = network linkage expansion.
def centrality(window, minp):
 out=pd.DataFrame(index=e.index,columns=A,dtype=float)
 for i,a in enumerate(A):
  vals=[]
  for b in A:
   if b!=a: vals.append(e[a].rolling(window,min_periods=minp).corr(e[b]))
  out[a]=pd.concat(vals,axis=1).mean(axis=1)
 return out
c60=centrality(60,42); c20=centrality(20,14)
f=c20-c60
print('FACTOR residual_correlation_network_centrality_expansion_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
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
for h in [10,20]:
 for name,mask in [('2020_24',ics[h].index<pd.Timestamp('2025')),('2025_26',(ics[h].index>=pd.Timestamp('2025'))&(ics[h].index<pd.Timestamp('2027'))),('2027_onward',ics[h].index>=pd.Timestamp('2027'))]:
  x=ics[h][mask];print('REGIME'+str(h),name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None,'hit',round((x>0).mean(),6))
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

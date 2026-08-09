"""Single candidate: residual defensive-basket correlation contraction, cutoff 2029-05-30."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2029-05-30')")
exec(prefix,globals())
# Interpretable factor: decline from structural to recent residual co-movement with the equal safe-haven basket.
defensive=e[['XAU','US10Y','CN10Y']].mean(axis=1)
c20=pd.DataFrame({a:e[a].rolling(20,min_periods=14).corr(defensive) for a in A})
c60=pd.DataFrame({a:e[a].rolling(60,min_periods=42).corr(defensive) for a in A})
f=c60-c20
print('FACTOR residual_defensive_basket_correlation_contraction_60_20d','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_reconstructed',len(lib))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q):out.append((t,q));ns.append(len(z))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_2024',ics[10].index<'2025'),('2025_2026',(ics[10].index>='2025')&(ics[10].index<'2027')),('2027_onward',ics[10].index>='2027')]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
s=[]
for n,v in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),v.stack().rename('v')],axis=1).dropna();rho=z.f.corr(z.v,method='spearman');s.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(s);print('MAX_ABS_LIBRARY_CORRELATION_RECONSTRUCTED',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))
print('Admission withheld unless correlation is subsequently verified against every currently admitted factor.')

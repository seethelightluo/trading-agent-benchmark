"""Validate one idea: VIX-stress-conditioned residual tail asymmetry (20d)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2029-07-11')")
exec(prefix,globals())
# Tail asymmetry is high when positive residual shocks dominate negative shocks.
# Interact it with *idiosyncratic* VIX-shock sensitivity: in stress-sensitive
# assets, resilience has a different economic meaning than generic skewness.
vixret=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].pct_change().reindex(p.index)
vbeta=pd.DataFrame({a:e[a].rolling(20,min_periods=14).cov(vixret)/(vixret.rolling(20,min_periods=14).var()+1e-12) for a in A})
up=e.clip(lower=0).pow(2).rolling(20,min_periods=14).mean().pow(.5)
down=(-e.clip(upper=0)).pow(2).rolling(20,min_periods=14).mean().pow(.5)
asym=(up-down)/(up+down+1e-12)
f=asym*(-vbeta)
print('FACTOR vix_stress_conditioned_residual_tail_asymmetry_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'reconstructed_library',len(lib))
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
for name,mask in [('2020_24',ics[10].index<'2025'),('2025_26',(ics[10].index>='2025')&(ics[10].index<'2027')),('2027_onward',ics[10].index>='2027')]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(z)))
if screen:
 mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))

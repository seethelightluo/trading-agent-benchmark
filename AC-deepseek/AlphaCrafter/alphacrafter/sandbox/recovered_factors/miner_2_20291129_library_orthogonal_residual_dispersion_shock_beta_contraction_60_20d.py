"""Validate one idea: residual dispersion-shock beta contraction (60d vs 20d)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290920_residual_common_shock_correlation_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# Candidate: residual correlation')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-09-05')", "END=pd.Timestamp('2029-11-28')")
exec(prefix,globals())
# Cross-asset dispersion shock is the daily standard deviation of residual returns.
# High signal: an asset's residual loading on dispersion shocks has contracted
# versus its 60d baseline, indicating a recent decoupling from fragmentation risk.
disp=e.std(axis=1,ddof=1)
def load(asset,w,minp):
 return e[asset].rolling(w,min_periods=minp).cov(disp)/disp.rolling(w,min_periods=minp).var()
raw=pd.DataFrame({a:load(a,60,42)-load(a,20,14) for a in A})
# Remove the nearest retained cross-sectional state-transition feature per date.
anchor='miner_1_residualized_defensive_correlation_decoupling_60_20'
def orth(t):
 z=pd.concat([raw.loc[t].rename('x'),lib[anchor].loc[t].rename('a')],axis=1).dropna()
 if len(z)<8 or z.a.nunique()<2:return pd.Series(index=A,dtype=float)
 beta=np.cov(z.x,z.a,ddof=1)[0,1]/np.var(z.a,ddof=1)
 return raw.loc[t]-beta*lib[anchor].loc[t]
f=pd.DataFrame({t:orth(t) for t in raw.index}).T
print('FACTOR library_orthogonal_residual_dispersion_shock_beta_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'anchor',anchor)
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
for name,mask in [('2020_24',ics[20].index<'2025'),('2025_26',(ics[20].index>='2025')&(ics[20].index<'2027')),('2027_28',(ics[20].index>='2027')&(ics[20].index<'2029')),('2029_ytd',ics[20].index>='2029')]:
 x=ics[20][mask];print('REGIME20',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turns=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
screen=[]
for name,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');screen.append((abs(rho),name,rho,len(z)))
mx,name,rho,cells=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',name,rho,'cells',cells)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))

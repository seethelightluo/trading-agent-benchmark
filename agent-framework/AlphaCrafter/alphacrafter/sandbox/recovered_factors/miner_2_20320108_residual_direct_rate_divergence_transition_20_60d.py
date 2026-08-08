"""Validate direct-yield-change US/CN divergence loading transition, one candidate."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_2_20310501_residual_vix_usdcny_joint_stress_loading_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# A continuous joint-risk driver:')[0].replace("END=pd.Timestamp('2031-04-30')","END=pd.Timestamp('2032-01-07')")
exec(prefix,globals())
# Yields are level series: use standardized first differences, never percentage returns.
def zychange(name):
 x=p[name].astype(float).diff()
 return x/(x.rolling(60,min_periods=40).std()+1e-12)
g=zychange('US10Y')-zychange('CN10Y')
# A high score means residual exposure to the widening rate gap has recently declined.
f=beta(e,g,60,42)-beta(e,g,20,14)
print('FACTOR residual_direct_rate_divergence_transition_loading_contraction_20_60d','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
print('DRIVER_NON_NULL',int(g.notna().sum()),'FACTOR_CELLS',int(f.notna().sum().sum()),'LAST500_UNIQUES',f.iloc[-500:].nunique(axis=1).value_counts().to_dict())
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for t in f.index[:-h]:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): out.append((t,q)); ns.append(len(z))
 z=pd.Series(dict(out),dtype=float); ics[h]=z; sd=z.std(ddof=1)
 d={'daily_paper_ic':z.mean(),'daily_paper_icir':z.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(z)),'ic_hit_ratio':(z>0).mean(),'ic_dates':len(z),'mean_valid_instruments':np.mean(ns)}; metrics[h]=d
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in d.items()}))
for name,mask in [('2020_24',ics[10].index<'2025-01-01'),('2025_26',(ics[10].index>='2025-01-01')&(ics[10].index<'2027-01-01')),('2027_onward',ics[10].index>='2027-01-01')]:
 z=ics[10][mask]; print('REGIME10',name,'DATES',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'HIT',round((z>0).mean(),6))
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(tos)),6),'TURNOVER_DATES',len(tos))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman')
 if pd.notna(rho): screen.append((abs(rho),n,rho,len(z)))
if screen:
 mx,n,rho,c=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',n,'RHO',round(rho,6),'CELLS',c)
else: print('MAX_ABS_LIBRARY_CORRELATION MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(d['daily_paper_ic']),6),'icir':round(float(d['daily_paper_icir']),6),'dates':d['ic_dates']} for h,d in metrics.items()}))

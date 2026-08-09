"""Validate one idea: percentile joint-weakness residual defensive beta transition."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_2_20290712_joint_equity_commodity_down_defensive_beta_transition_60_20d.py',encoding='utf8').read()
prefix=src.split('# Candidate: change in residual beta')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-07-11')", "END=pd.Timestamp('2029-07-25')")
exec(prefix,globals())
# Candidate: transition in defensive-residual beta measured when equities are weak
# and commodity losses are in their trailing 60-session lower 40% tail. This
# relaxes the simultaneous-negative rule while retaining a growth-stress meaning.
defensive=e[['XAU','US10Y','CN10Y']].mean(axis=1)
equity=r[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1)
commodity=r[['COPPER','WTI']].mean(axis=1)
ctail=commodity.rolling(60,min_periods=40).quantile(.40).shift(1)
stress=(equity<0)&(commodity<ctail)
def rbeta_stress(w,n):
 d=defensive.where(stress)
 return pd.DataFrame({a:e[a].where(stress).rolling(w,min_periods=n).cov(d)/(d.rolling(w,min_periods=n).var()+1e-12) for a in A})
f=rbeta_stress(60,15)-rbeta_stress(20,5)
print('FACTOR percentile_joint_weakness_residual_defensive_basket_beta_transition_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'reconstructed_library',len(lib),'stress_fraction',round(float(stress.mean()),6))
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
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman');screen.append((abs(rho),name,rho,len(z)))
mx,name,rho,cells=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',name,'rho',round(float(rho),6),'cells',cells)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))

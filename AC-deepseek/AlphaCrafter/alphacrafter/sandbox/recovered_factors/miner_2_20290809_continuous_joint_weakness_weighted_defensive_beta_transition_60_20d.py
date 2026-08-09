"""Validate one idea: continuous joint-weakness-weighted residual defensive beta transition."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_2_20290628_downside_state_residual_defensive_beta_transition_60_20d.py',encoding='utf8').read()
prefix=src.split('# Candidate: beta transition')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-06-27')", "END=pd.Timestamp('2029-08-08')")
exec(prefix,globals())
# Candidate: change in beta to the defensive residual basket, using all sessions
# but smoothly upweighting simultaneous equity and commodity weakness. Weight is
# 1 plus clipped negative standardized returns, avoiding sparse binary states.
defensive=e[['XAU','US10Y','CN10Y']].mean(axis=1)
equity=r[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1)
commodity=r[['COPPER','WTI']].mean(axis=1)
eqz=equity/(equity.rolling(60,min_periods=40).std().shift(1)+1e-12)
coz=commodity/(commodity.rolling(60,min_periods=40).std().shift(1)+1e-12)
w=1+(-eqz).clip(0,3)+(-coz).clip(0,3)
def wbeta(x,y,weight,window):
 # weighted covariance / weighted variance, all terms only through date t
 sw=weight.rolling(window,min_periods=window//2).sum()
 mx=(weight*x).rolling(window,min_periods=window//2).sum()/sw
 my=(weight*y).rolling(window,min_periods=window//2).sum()/sw
 cov=(weight*x*y).rolling(window,min_periods=window//2).sum()/sw-mx*my
 var=(weight*y*y).rolling(window,min_periods=window//2).sum()/sw-my*my
 return cov/(var+1e-12)
f=pd.DataFrame({a:wbeta(e[a],defensive,w,60)-wbeta(e[a],defensive,w,20) for a in A})
print('FACTOR continuous_joint_weakness_weighted_residual_defensive_basket_beta_transition_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'reconstructed_library',len(lib),'mean_weight',round(float(w.mean()),6),'p95_weight',round(float(w.quantile(.95)),6))
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

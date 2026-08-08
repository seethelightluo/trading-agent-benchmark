"""One candidate: residual continuous US-CN yield-level-spread innovation loading contraction.
As-of 2031-08-20; outer daily panel is forward-filled *only from prior observed closes*,
so holiday fills are known as-of each session and the post-2026 synthetic period is retained.
"""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20310123_positive_usdjpy_change_shock_loading_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# Candidate: sensitivity to directly observable *positive* yen')[0].replace("END=pd.Timestamp('2031-01-22')","END=pd.Timestamp('2031-08-20')")
exec(prefix,globals())
# Add post-Jan-2031 admitted signal definitions to the inherited exhaustive reconstruction.
# All are evaluated on the same as-of aligned panel; these additions make the screen conservative.
# Return-autocorrelation expansion and jump-concentration expansions.
lib['miner_1_residual_return_autocorrelation_expansion_20_60d']=pd.DataFrame({a:e[a].rolling(20,min_periods=14).corr(e[a].shift(1))-e[a].rolling(60,min_periods=42).corr(e[a].shift(1)) for a in A})
jump=(e.abs()>2*e.rolling(60,min_periods=40).std()).astype(float)
lib['miner_1_residual_jump_concentration_expansion_20_60d']=jump.rolling(20,min_periods=14).mean()-jump.rolling(60,min_periods=42).mean()
posjump=((e>2*e.rolling(60,min_periods=40).std())).astype(float)
lib['miner_1_residual_positive_jump_concentration_expansion_20_60d']=posjump.rolling(20,min_periods=14).mean()-posjump.rolling(60,min_periods=42).mean()
# Candidate driver: daily innovation in yield level differential, standardized trailing-only.
yspread=p['US10Y']-p['CN10Y']; dy=yspread.diff()
driver=((dy-dy.rolling(60,min_periods=40).mean())/(dy.rolling(60,min_periods=40).std()+1e-12)).clip(-5,5).fillna(0)
def rollbeta2(x,w,n): return pd.DataFrame({a:e[a].rolling(w,min_periods=n).cov(x)/(x.rolling(w,min_periods=n).var()+1e-12) for a in A})
f=rollbeta2(driver,60,42)-rollbeta2(driver,20,14)
print('FACTOR residual_continuous_us_cn_yield_level_spread_innovation_loading_contraction_20_60d','end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'sessions',len(p),'universe',len(A),'library',len(lib),'driver_nonzero',round(float((driver!=0).mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z):out.append((t,z));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to),'VALID_CELLS',int(f.notna().sum().sum()))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(q)))
mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))

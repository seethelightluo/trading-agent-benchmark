"""One candidate: residual continuous risk-transition loading contraction (20d vs 60d)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_2_20310501_residual_vix_usdcny_joint_stress_loading_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# A continuous joint-risk driver:')[0].replace("END=pd.Timestamp('2031-04-30')","END=pd.Timestamp('2031-09-03')")
exec(prefix,globals())
# Continuous global risk-transition driver: standardized VIX and DXY shocks,
# offset by the contemporaneous broad-equity return.  It is non-sparse and uses
# only daily completed macro observations.  The score is a recent-minus-slow
# idiosyncratic loading, signed as contraction (slow minus recent).
def zret_obs(name):
 x=pd.read_csv('../persistent/index_data/'+name+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change().reindex(p.index)
 return x/(x.rolling(60,min_periods=40).std()+1e-12)
driver=zret_obs('VIX')+zret_obs('DXY')-r['SPX']/(r['SPX'].rolling(60,min_periods=40).std()+1e-12)
f=beta(e,driver,60,42)-beta(e,driver,20,14)
print('FACTOR residual_continuous_global_risk_transition_loading_contraction_20_60d','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'driver_nonnull',round(driver.notna().mean(),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): out.append((t,q)); ns.append(len(z))
 z=pd.Series(dict(out),dtype=float); ics[h]=z; ss=z.std(ddof=1)
 metrics[h]={'daily_paper_ic':z.mean(),'daily_paper_icir':z.mean()/ss,'ic_std':ss,'ic_standard_error':ss/np.sqrt(len(z)),'ic_hit_ratio':(z>0).mean(),'ic_dates':len(z),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 z=ics[10][mask]; print('REGIME10',name,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
rk=f.rank(axis=1,pct=True); to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman'); screen.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',n,'rho',round(rho,6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']}for h,q in metrics.items()}))

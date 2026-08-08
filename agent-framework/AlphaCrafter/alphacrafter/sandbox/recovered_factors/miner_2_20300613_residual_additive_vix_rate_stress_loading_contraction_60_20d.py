"""One candidate: residual loading transition to additive VIX--rate stress (20d vs 60d)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0].replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2030-06-12')")
exec(prefix,globals())
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).reindex(p.index).ffill()
# Additive state avoids the degenerate coincidence requirement of multiplying two shocks.
vchg=vix.pct_change(); rchg=p['US10Y'].pct_change()
vpos=(vchg/(vchg.rolling(60,min_periods=40).std()+1e-12)).clip(lower=0,upper=5)
rabs=(rchg.abs()/(rchg.abs().rolling(60,min_periods=40).std()+1e-12)).clip(upper=5)
stress=(vpos+rabs).replace([np.inf,-np.inf],np.nan)
def loading(a,w,n):
 return e[a].rolling(w,min_periods=n).cov(stress)/(stress.rolling(w,min_periods=n).std()+1e-12)
f=pd.DataFrame({a:loading(a,60,42)-loading(a,20,14) for a in A})
print('FACTOR residual_additive_vix_rate_stress_loading_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'reconstructed_library',len(lib),'stress_nonzero_fraction',round(float((stress.fillna(0)>0).mean()),6),'stress_std',round(float(stress.std()),6),'factor_cells',int(f.notna().sum().sum()))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): out.append((t,q));ns.append(len(z))
 x=pd.Series([q for t,q in out],index=pd.DatetimeIndex([t for t,q in out]));ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');screen.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(screen);print('LIBRARY_SCREEN',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))

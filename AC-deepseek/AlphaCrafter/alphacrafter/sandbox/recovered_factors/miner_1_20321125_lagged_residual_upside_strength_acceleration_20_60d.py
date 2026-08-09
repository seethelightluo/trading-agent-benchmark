"""One candidate: lagged residual upside-strength acceleration (20d vs 60d).
Uses continuous positive idiosyncratic returns, scaled by each asset's 60d residual
volatility: recent residual upside contribution minus its quarterly baseline.
All signal inputs are lagged one completed day; forward returns are labels only.
"""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20310123_positive_usdjpy_change_shock_loading_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# Candidate: sensitivity to directly observable')[0].replace("END=pd.Timestamp('2031-01-22')","END=pd.Timestamp('2032-11-24')")
exec(prefix,globals())
# Continuous analogue of residual positive-participation acceleration.  Scaling
# makes a 1% residual move comparable across crypto, equities, yields and metals.
up=e.clip(lower=0)
scale=e.rolling(60,min_periods=42).std()+1e-12
f=(up.rolling(20,min_periods=14).mean()-up.rolling(60,min_periods=42).mean()).div(scale).shift(1)
print('FACTOR lagged_residual_upside_strength_acceleration_20_60d')
print('VALIDATION_END',END.date(),'CALENDAR_DATES',len(p),'UNIVERSE',len(A),'RECONSTRUCTED_LIBRARY',len(lib))
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p)-1;out=[];ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z):out.append((t,z));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_dates':len(x),'hit_ratio':(x>0).mean(),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_2024',ics[10].index<'2025-01-01'),('2025_2026',(ics[10].index>='2025-01-01')&(ics[10].index<'2027-01-01')),('2027_onward',ics[10].index>='2027-01-01')]:
 x=ics[10][mask];print('REGIME_10D',name,'dates',len(x),'ic',round(x.mean(),6),'icir',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(q)))
if screen:
 mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION EVIDENCE_MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))

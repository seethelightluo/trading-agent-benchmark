"""One-factor screen: negative USDJPY change shock residual loading expansion (60d vs 20d).
A yen-strengthening shock is an observable cross-asset risk-off impulse.  The signal is each
asset's residual loading on that impulse, recently increased versus its structural loading.
"""
import json,numpy as np,pandas as pd
# Reuse the audited price panel/residual construction and reconstructed contemporaneous library.
src=open('scripts/miner_1_20310123_positive_usdjpy_change_shock_loading_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# Candidate: sensitivity to directly observable *positive* yen')[0].replace("END=pd.Timestamp('2031-01-22')","END=pd.Timestamp('2032-04-28')")
exec(prefix,globals())
# Candidate only: magnitude of directly observed daily yen-strengthening (negative USDJPY) shock.
fx=pd.read_csv('../persistent/index_data/USDJPY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).reindex(p.index).ffill().pct_change()
shock=(-fx.clip(upper=0)).fillna(0)
f=rollbeta(shock,20,14)-rollbeta(shock,60,42)
print('FACTOR residual_negative_usdjpy_change_shock_loading_expansion_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'reconstructed_library',len(lib),'shock_nonzero_fraction',round(float((shock>0).mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1); out=[]; ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v): out.append((t,v));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_dates':len(x),'hit_ratio':(x>0).mean(),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_2024',ics[10].index<pd.Timestamp('2025-01-01')),('2025_2026',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 x=ics[10][mask];print('REGIME_10D',name,'dates',len(x),'ic',round(float(x.mean()),6),'icir',round(float(x.mean()/x.std(ddof=1)),6),'hit_ratio',round(float((x>0).mean()),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna()
 rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(q)))
if screen:
 mx,n,rho,c=max(screen);print('RECONSTRUCTED_LIBRARY_MAX_ABS_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
else:print('RECONSTRUCTED_LIBRARY_CORRELATION_EVIDENCE_MISSING')
print('DECAY',json.dumps({str(h):round(float(x['daily_paper_ic']),6) for h,x in metrics.items()}))
print('ADMISSION_NOTE: reconstructed-library screen is diagnostic only; binding admission additionally requires exact signals for every currently admitted factor.')

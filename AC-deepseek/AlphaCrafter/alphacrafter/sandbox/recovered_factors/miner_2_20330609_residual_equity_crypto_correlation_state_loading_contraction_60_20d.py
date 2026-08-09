"""One candidate: residual equity-crypto correlation-state loading contraction (60d/20d), repaired standalone loader."""
import json,numpy as np,pandas as pd
# Use the known self-contained historical loader directly; do not inherit nested cutoff substitutions.
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')","END=pd.Timestamp('2033-06-08')")
exec(prefix,globals())
# Completed-day 20d SPX/BTC linkage change, normalized over 60d; signal is long-minus-short residual beta.
eqcrypto=r['SPX'].rolling(20,min_periods=15).corr(r['BTC'])
delta=eqcrypto.diff(); driver=((delta-delta.rolling(60,min_periods=42).mean())/(delta.rolling(60,min_periods=42).std()+1e-12)).clip(-6,6)
def rollbeta(driver,w,n): return pd.DataFrame({a:e[a].rolling(w,min_periods=n).cov(driver)/(driver.rolling(w,min_periods=n).var()+1e-12) for a in A})
f=rollbeta(driver,60,42)-rollbeta(driver,20,14)
print('FACTOR residual_equity_crypto_correlation_state_loading_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_snapshot',len(lib),'driver_coverage',round(float(driver.notna().mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z):out.append((t,z));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float); ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask]; print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to),'VALID_CELLS',int(f.notna().sum().sum()))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(q)))
if screen:
 mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION_SNAPSHOT',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION_SNAPSHOT EVIDENCE_MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))
print('NOTE library snapshot is historical and therefore not sufficient for admission against current active library.')

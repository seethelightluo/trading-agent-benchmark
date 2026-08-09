"""One candidate: residual gold–US-rate relative-innovation loading expansion, 20d vs 60d.
Tests a continuous defensive/rates transition driver through the last completed bar."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20310123_positive_usdjpy_change_shock_loading_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# Candidate: sensitivity to directly observable *positive* yen')[0].replace("END=pd.Timestamp('2031-01-22')","END=pd.Timestamp('2031-07-09')")
exec(prefix,globals())
# Driver is the standardized daily relative return of gold to the US 10-year
# yield series. It is continuous, unlike the prior sparse yield-spread surprise.
relative=r['XAU']-r['US10Y']
driver=((relative-relative.rolling(60,min_periods=40).mean())/(relative.rolling(60,min_periods=40).std()+1e-12)).clip(-5,5).fillna(0.0)
# Higher factor = sensitivity of residual asset return to this defensive-relative
# innovation has expanded in the recent 20 sessions versus its 60-session level.
f=rollbeta(driver,20,14)-rollbeta(driver,60,42)
print('FACTOR residual_gold_us10y_relative_innovation_loading_expansion_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'driver_nonzero_fraction',round(float((driver!=0).mean()),6),'driver_std',round(float(driver.std()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v):out.append((t,v));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns),'VALID_CELLS',int(f.notna().sum().sum()))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(q)))
if screen:
 mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))

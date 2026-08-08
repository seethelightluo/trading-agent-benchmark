"""Miner_2 one-idea validation: cross-asset breadth-imbalance shock loading contraction.
The observable driver is the daily fraction of the 15 traded assets with a positive
return minus fraction with a negative return, standardized over 60 completed days.
The factor is each asset's 60d-minus-20d beta to this broad participation shock.
Higher values mean recent breadth-imbalance sensitivity contracted."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_2_20330203_residual_broad_dollar_innovation_loading_contraction_60_20d.py',encoding='utf8').read()
prefix=src.split('# Dollar-strength impulse:')[0].replace("END=pd.Timestamp('2033-02-02')","END=pd.Timestamp('2033-02-16')")
exec(prefix,globals())
r=p.pct_change()
# Sign breadth is a unit-free, cross-asset participation measure. Zero-return series
# do not contribute to either side, avoiding arbitrary classification of flat yields.
imbalance=(r.gt(0).sum(axis=1)-r.lt(0).sum(axis=1))/r.notna().sum(axis=1)
driver=((imbalance-imbalance.rolling(60,min_periods=42).mean())/(imbalance.rolling(60,min_periods=42).std()+1e-12)).clip(-6,6)
f=rollbeta(driver,60,42)-rollbeta(driver,20,14)
print('FACTOR cross_asset_breadth_imbalance_shock_loading_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'driver_coverage',round(float(driver.notna().mean()),6))
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
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None,'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn),'VALID_CELLS',int(f.notna().sum().sum()))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),n,rho,len(q)))
if screen:
 mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION EVIDENCE_MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))
# Persist a signal artifact only for correlation reproducibility; admission depends on printed screening.
f.to_pickle('scripts/miner_2_20330217_cross_asset_breadth_imbalance_shock_loading_contraction_60_20d_signal.pkl')

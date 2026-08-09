"""One candidate: direct DXY strength-shock response contraction (60d vs 20d)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2030-04-17')")
exec(prefix,globals())
# A continuous USD-strength shock proxy: positive DXY return standardized by its
# completed 60-day variability.  The factor captures each tradable asset's recent
# reduction in adverse USD-shock loading relative to its structural loading.
dxy=pd.read_csv('../persistent/index_data/DXY.csv')
datecol=next(c for c in dxy.columns if c.lower() in ('date','datetime','trade_date'))
closecol=next(c for c in dxy.columns if c.lower() in ('close','adj_close','price'))
dxy[datecol]=pd.to_datetime(dxy[datecol]); dx=dxy.set_index(datecol)[closecol].astype(float).sort_index().reindex(p.index).ffill()
dxr=dx.pct_change(); shock=(dxr/(dxr.rolling(60,min_periods=40).std()+1e-12)).clip(lower=0)
pa=p.ffill(); ra=pa.pct_change()
c20=pd.DataFrame({a:ra[a].rolling(20,min_periods=14).corr(shock) for a in A})
c60=pd.DataFrame({a:ra[a].rolling(60,min_periods=42).corr(shock) for a in A})
f=c60-c20
print('FACTOR dxy_strength_shock_response_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'shock_positive_fraction',round(float((shock>0).mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=pa.shift(-h)/pa-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q): out.append((t,q)); ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 x=ics[10][mask]; print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman')
 if pd.notna(rho): screen.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))

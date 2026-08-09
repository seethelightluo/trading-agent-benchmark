"""One candidate: residual broad-breadth recovery loading expansion (20 versus 60 sessions)."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0]
prefix=prefix.replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2030-08-07')")
exec(prefix,globals())
# Breadth is the daily fraction of the 15 assets with positive return.  Its positive
# surprise isolates broad participation recoveries rather than a directional market return.
breadth=(p.pct_change()>0).mean(axis=1)
bshock=((breadth-breadth.rolling(60,min_periods=40).mean())/(breadth.rolling(60,min_periods=40).std()+1e-12)).clip(-5,5).clip(lower=0).fillna(0.)
def expo(a,w,n): return e[a].rolling(w,min_periods=n).cov(bshock)/(bshock.rolling(w,min_periods=n).std()+1e-12)
f=pd.DataFrame({a:expo(a,20,14)-expo(a,60,42) for a in A})
print('FACTOR residual_broad_breadth_recovery_loading_expansion_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'positive_breadth_shock_fraction',round(float((bshock!=0).mean()),6))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q):out.append((t,q));ns.append(len(z))
 x=pd.Series([q for t,q in out],index=pd.DatetimeIndex([t for t,q in out]));ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turns=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
screen=[]
for n,s in sorted(lib.items()):
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=z.f.corr(z.s,method='spearman');screen.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(q['daily_paper_ic']),6),'icir':round(float(q['daily_paper_icir']),6),'dates':q['ic_dates']} for h,q in metrics.items()}))

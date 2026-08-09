"""One candidate: residual loading transition to negative CN10Y-change shocks, fully library screened."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_3_20290222_residual_defensive_basket_correlation_contraction_research.py',encoding='utf8').read()
prefix=src.split('# Candidate: recent contraction versus structural correlation of residual returns with defensive basket.')[0].replace("END=pd.Timestamp('2029-02-21')", "END=pd.Timestamp('2030-06-26')")
exec(prefix,globals())
# State is a distinctly China-rate-specific tail event: negative daily CN10Y moves exceeding its own rolling median magnitude.
y=p['CN10Y'].pct_change(); thresh=y.abs().rolling(60,min_periods=40).median()
state=y.where((y<0)&(y.abs()>thresh),0.0)
def load(a,w,n): return e[a].rolling(w,min_periods=n).cov(state)/(state.rolling(w,min_periods=n).std()+1e-12)
# Positive score: sensitivity has contracted over the recent window relative to structural exposure.
f=pd.DataFrame({a:load(a,60,42)-load(a,20,14) for a in A})
print('FACTOR residual_negative_cn10y_shock_loading_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'shock_fraction',round(float((state!=0).mean()),6),'coverage',round(float(f.notna().mean().mean()),6))
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[]; ns=[]
 for t in f.index:
  z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=z.f.corr(z.y,method='spearman')
   if pd.notna(q):out.append((t,q));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'ic':x.mean(),'icir':x.mean()/sd,'hit':(x>0).mean(),'dates':len(x),'mean_n':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:(round(float(v),6) if k!='dates' else int(v)) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8:turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn))
screen=[]
for n,s in lib.items():
 z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); q=z.f.corr(z.s,method='spearman')
 if pd.notna(q):screen.append((abs(q),n,q,len(z)))
mx,n,q,c=max(screen);print('LIBRARY_SCREEN max_abs_library_correlation',round(float(mx),6),'factor',n,'rho',round(float(q),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(v['ic']),6),'icir':round(float(v['icir']),6),'dates':v['dates']} for h,v in metrics.items()}))

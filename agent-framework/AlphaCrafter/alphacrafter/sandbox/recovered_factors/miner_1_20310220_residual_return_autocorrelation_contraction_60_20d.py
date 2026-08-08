"""One candidate: residual return-autocorrelation contraction (60d versus 20d)."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20310206_us_cn_yield_spread_shock_loading_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# Candidate: changing asset sensitivity')[0].replace("END=pd.Timestamp('2031-02-05')","END=pd.Timestamp('2031-02-19')")
exec(prefix,globals())
# Candidate: each asset's own residual-return persistence, measured as lag-one
# autocorrelation after removing the equal-weight cross-asset market return.
# The signal is the structural (60d) persistence minus recent (20d) persistence:
# positive readings identify assets whose serial dependence has recently weakened.
def ac1(x,w,n):
    return x.rolling(w,min_periods=n).corr(x.shift(1))
f=pd.DataFrame({a:ac1(e[a],60,42)-ac1(e[a],20,14) for a in A})
print('FACTOR residual_return_autocorrelation_contraction_60_20d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1; out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v):out.append((t,v));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025')),('2025_26',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask]; print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=q.f.corr(q.s,method='spearman'); screen.append((abs(rho),n,rho,len(q)))
mx,n,rho,c=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']}for h,v in metrics.items()}))

"""One candidate: residual downside-versus-upside persistence asymmetry transition.
Uses only completed observations through 2031-09-03."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20310724_revalidate_residual_return_autocorrelation_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# Add the signals admitted')[0].replace("END=pd.Timestamp('2031-07-23')","END=pd.Timestamp('2031-09-03')")
exec(prefix,globals())
# Reconstruct the two subsequent Miner-1 admitted signals for the correlation
# screen. The inherited library contains the other currently admitted signals.
def jump_share(x,w,n,side=None):
 def calc(z):
  z=np.asarray(z,float)
  a=np.abs(z) if side is None else (np.maximum(z,0) if side=='pos' else np.maximum(-z,0))
  k=max(1,int(np.ceil(.2*len(a))))
  return np.partition(a,-k)[-k:].sum()/a.sum() if a.sum()>0 else np.nan
 return x.rolling(w,min_periods=n).apply(calc,raw=True)
lib['miner_1_residual_jump_concentration_expansion_20_60d']=pd.DataFrame({a:jump_share(e[a],20,14)-jump_share(e[a],60,42) for a in A})
lib['miner_1_residual_positive_jump_concentration_expansion_20_60d']=pd.DataFrame({a:jump_share(e[a],20,14,'pos')-jump_share(e[a],60,42,'pos') for a in A})
# Difference of lag-one serial dependence conditional on negative versus
# positive residual sessions, then recent (20d) minus structural (60d) change.
def conditional_ac(x, side, w, minp):
 z=x.where(x<0) if side=='down' else x.where(x>0)
 return z.rolling(w,min_periods=minp).apply(lambda q: pd.Series(q).autocorr(1),raw=True)
f=pd.DataFrame({a:(conditional_ac(e[a],'down',20,14)-conditional_ac(e[a],'down',60,42))-(conditional_ac(e[a],'up',20,14)-conditional_ac(e[a],'up',60,42)) for a in A})
print('FACTOR residual_downside_vs_upside_autocorrelation_asymmetry_transition_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library_screened',len(lib))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v): out.append((t,v));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
idx=pd.to_datetime(ics[10].index)
for name,mask in [('2020_24',idx<pd.Timestamp('2025')),('2025_26',(idx>=pd.Timestamp('2025'))&(idx<pd.Timestamp('2027'))),('2027_onward',idx>=pd.Timestamp('2027'))]:
 x=ics[10][mask];print('REGIME10',name,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(to)),6),'TURNOVER_DATES',len(to))
screen=[]
for n,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho): screen.append((abs(rho),n,rho,len(q)))
if screen:
 mx,n,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']} for h,v in metrics.items()}))

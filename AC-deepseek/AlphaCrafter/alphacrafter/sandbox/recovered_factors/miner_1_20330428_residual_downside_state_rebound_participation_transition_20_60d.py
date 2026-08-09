"""One candidate: residual downside-state rebound participation transition, 20d vs 60d.
For each asset, measures the fraction of its positive residual days which occur one day
after a negative equal-weight-market day, normalized by opportunities.  The short-minus-
long transition identifies changing idiosyncratic rebound participation during broad risk-off
sessions using only fully available price data."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20311127_residual_downside_tail_severity_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# For each window')[0].replace("END=pd.Timestamp('2031-11-26')","END=pd.Timestamp('2033-04-27')")
exec(prefix,globals())
# Market state is lagged by one completed day relative to each residual return,
# so each factor observation only uses information through its own date.
mkt=e.mean(axis=1)
state=(mkt.shift(1)<0).astype(float)
def rebound_participation(x,w,n):
    # conditional mean of positive-residual indicator across market-down opportunities
    numerator=((x>0).astype(float)*state).rolling(w,min_periods=n).sum()
    denom=state.rolling(w,min_periods=n).sum()
    return numerator/denom.replace(0,np.nan)
f=pd.DataFrame({a:rebound_participation(e[a],20,14)-rebound_participation(e[a],60,42) for a in A})
print('FACTOR residual_downside_state_rebound_participation_transition_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
metrics={};ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h)/p-1;out=[];ns=[]
 for t in f.index:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if pd.notna(v):out.append((t,v));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x; sd=x.std(ddof=1)
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
for name,s in sorted(lib.items()):
 q=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna();rho=q.f.corr(q.s,method='spearman')
 if pd.notna(rho):screen.append((abs(rho),name,rho,len(q)))
if screen:
 mx,name,rho,c=max(screen);print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',name,'rho',round(float(rho),6),'cells',c)
else: print('MAX_ABS_LIBRARY_CORRELATION EVIDENCE_MISSING')
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':v['ic_dates']} for h,v in metrics.items()}))
f.to_pickle('scripts/miner_1_20330428_residual_downside_state_rebound_participation_transition_20_60d_signal.pkl')

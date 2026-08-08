"""One candidate: residual downside-run contraction (20d versus 60d).
For each asset, form market-residual daily returns.  On each date, measure the
average length of consecutive negative-residual-return runs that *ended* in each
lookback window.  The signal is the negative change (20d minus 60d): higher
values mean downside spells have recently become shorter, an interpretable
stress-conditioned recovery-speed / path-shape measure independent of return
magnitude."""
import json,numpy as np,pandas as pd
src=open('scripts/miner_1_20311127_residual_downside_tail_severity_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# For each window')[0].replace("END=pd.Timestamp('2031-11-26')","END=pd.Timestamp('2033-05-25')")
exec(prefix,globals())
def ended_run_length(x):
    neg=x.lt(0)
    groups=(neg.ne(neg.shift())).cumsum()
    length=neg.groupby(groups).transform('sum').where(neg,0.0)
    # A run is observable at t only after it ends; assign its known length to end day.
    return length.shift(1).where((~neg)&neg.shift(1).fillna(False))
def rolling_ended_mean(x,w,minp):
    return ended_run_length(x).rolling(w,min_periods=minp).mean()
f=pd.DataFrame({a:-(rolling_ended_mean(e[a],20,4)-rolling_ended_mean(e[a],60,12)) for a in A})
print('FACTOR residual_downside_run_contraction_20_60d validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib))
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
f.to_pickle('scripts/miner_1_20330526_residual_downside_run_contraction_20_60d_signal.pkl')

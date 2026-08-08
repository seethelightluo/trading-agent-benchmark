"""Timing-safe one-factor validation: inverse residual drawdown-dispersion-conditioned recovery deterioration."""
import numpy as np, pandas as pd, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-06-23'); cal=pd.bdate_range('2020-01-01',END)
def close(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float)
p=pd.DataFrame({a:close(a).reindex(cal).ffill() for a in A}); r=p.pct_change(); m=r.mean(axis=1)
b=pd.DataFrame({a:r[a].rolling(60,min_periods=42).cov(m)/(m.rolling(60,min_periods=42).var()+1e-12) for a in A}); e=r-b.mul(m,axis=0)
# At t, completed paired observation is (e[t-1],e[t]). No e[t+1] is used.
def recovery_safe(x,w,n):
    def fun(v):
        v=np.asarray(v,float); event=v[:-1]; outcome=v[1:]
        hist=event[np.isfinite(event)]; sd=np.std(hist,ddof=1) if len(hist)>1 else np.nan
        take=(event < -sd) & np.isfinite(outcome)
        return np.mean(outcome[take])/sd if np.isfinite(sd) and sd>1e-12 and take.any() else np.nan
    return x.rolling(w,min_periods=n).apply(fun,raw=True)
rec20=pd.DataFrame({a:recovery_safe(e[a],20,14) for a in A}); rec60=pd.DataFrame({a:recovery_safe(e[a],60,42) for a in A})
level=(1+e.fillna(0)).cumprod(); dd=level.div(level.rolling(20,min_periods=14).max()).sub(1)
scale=(dd.std(axis=1)/dd.std(axis=1).rolling(60,min_periods=42).median()).clip(.5,2)
f=-(rec20-rec60).mul(scale,axis=0)
print('FACTOR inverse_residual_drawdown_dispersion_conditioned_recovery_deterioration_20_60d_TIMING_SAFE')
print('TIMING_AUDIT factor_at_t_uses_only_residual_pairs_(t-1,t)_and_earlier; forward_return_starts_after_t')
print('VALIDATION_END',END.date(),'CALENDAR_DATES',len(cal),'UNIVERSE',len(A))
ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1); out=[]; ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z):out.append((t,z));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x; sd=x.std(ddof=1)
 print('HORIZON',h,json.dumps({'daily_paper_ic':round(x.mean(),6),'daily_paper_icir':round(x.mean()/sd,6),'ic_standard_error':round(sd/np.sqrt(len(x)),6),'ic_dates':len(x),'hit_ratio':round((x>0).mean(),6),'mean_valid_instruments':round(float(np.mean(ns)),4)}))
for name,mask in [('2020_2024',ics[10].index<pd.Timestamp('2025')),('2025_2026',(ics[10].index>=pd.Timestamp('2025'))&(ics[10].index<pd.Timestamp('2027'))),('2027_onward',ics[10].index>=pd.Timestamp('2027'))]:
 x=ics[10][mask];print('REGIME_10D',name,'dates',len(x),'ic',round(x.mean(),6),'icir',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
# Exact timing-safe reconstruction of nearest admitted factor.
def old_recovery(x,w,n): return recovery_safe(x,w,n)
low20=(-e.clip(upper=0)).rolling(20,min_periods=14).std();rel=(low20/low20.median(axis=1).replace(0,np.nan)).clip(.25,4)
sibling=-(rec20-rec60).mul(rel,axis=0)
q=pd.concat([f.stack().rename('candidate'),sibling.stack().rename('sibling')],axis=1).dropna()
rho=q.candidate.corr(q.sibling,method='spearman')
print('RELATED_ADMITTED_CORRELATION',json.dumps({'factor':'miner_1_20320429_relative_downside_risk_conditioned_inverse_residual_recovery_expansion_20_60d','rho':round(rho,6),'common_cells':len(q)}))
print('DECAY',json.dumps({str(h):round(float(x.mean()),6) for h,x in ics.items()}))
print('FULL_LIBRARY_STATUS_NOT_ADMISSIBLE_UNLESS_ALL_ADMITTED_SIGNALS_ARE_RECONSTRUCTED')

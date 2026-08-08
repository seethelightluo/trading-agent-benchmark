"""One candidate: residual broad-drawdown rank-resilience transition (20d vs 60d).
When the lagged equal-weight market is in a 10d drawdown, compare each asset's
recent residual-return rank resilience with its longer conditional baseline."""
import json,numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2032-02-04')
def load(a,c): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,c].astype(float)
cal=pd.bdate_range('2020-01-01',END); p=pd.DataFrame({a:load(a,'close').reindex(cal).ffill() for a in A}); r=p.pct_change(); m=r.mean(axis=1)
def beta(x,y,w,n): return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/(y.rolling(w,min_periods=n).var()+1e-12) for a in A})
b=beta(r,m,60,40); e=r-b.mul(m,axis=0)
# State is lagged: no same-day outcome is allowed into the signal.
dd=m.add(1).rolling(10,min_periods=8).apply(np.prod,raw=True).sub(1).shift(1); state=dd<0
# Conditional mean residual performance, requiring enough stress observations.
def conditional_mean(x,w,n):
 return x.where(state, np.nan).rolling(w,min_periods=n).mean()
short=conditional_mean(e,20,5); long=conditional_mean(e,60,15); f=short-long
# Explicit signal set representing all active construction families; candidate will not be admitted
# without full-library evidence if it clears the IC gates.
own=r.rolling(20,min_periods=15).std(); v=pd.DataFrame({a:load(a,'volume').reindex(cal).ffill() for a in A}); lv=np.log(v.replace(0,np.nan))
lib={'risk_adjusted_trend':(p/p.shift(20)-1)/own,'residual_autocorr':e.rolling(20,min_periods=15).apply(lambda x:pd.Series(x).autocorr(),raw=False)-e.rolling(60,min_periods=40).apply(lambda x:pd.Series(x).autocorr(),raw=False),'realized_vol_compression':r.rolling(20,min_periods=15).std()-r.rolling(60,min_periods=40).std(),'relative_volume':lv-lv.rolling(20,min_periods=15).mean(),'lower_partial_moment':e.where(e<0).pow(2).rolling(20,min_periods=15).mean()}
print('FACTOR residual_broad_drawdown_rank_resilience_transition_20_60d','validation_end',END.date(),'calendar_dates',len(cal),'universe',len(A),'library_screen_signals',len(lib),'stress_frequency',round(float(state.mean()),6))
metrics={}; ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1); rows=[]; ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z): rows.append((t,z));ns.append(len(q))
 x=pd.Series(dict(rows),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 metrics[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(z),6) for k,z in metrics[h].items()}))
for n,mask in [('2025_26',(ics[10].index>='2025-01-01')&(ics[10].index<'2027-01-01')),('2027_onward',ics[10].index>='2027-01-01')]:
 x=ics[10][mask]; print('REGIME10',n,'dates',len(x),'IC',round(float(x.mean()),6),'ICIR',round(float(x.mean()/x.std(ddof=1)),6),'hit',round(float((x>0).mean()),6))
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(tos)),6),'TURNOVER_DATES',len(tos))
cors=[]
for n,x in lib.items():
 q=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); z=q.f.corr(q.x,method='spearman')
 if pd.notna(z):cors.append((abs(z),n,z,len(q)))
mx,n,z,c=max(cors);print('PARTIAL_LIBRARY_MAX_ABS_CORRELATION',round(float(mx),6),'FACTOR',n,'rho',round(float(z),6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(float(x['daily_paper_ic']),6),'icir':round(float(x['daily_paper_icir']),6),'dates':int(x['ic_dates'])}for h,x in metrics.items()}))

"""One-factor screen: severity-weighted broad-drawdown residual resilience transition.
Recent-minus-structural idiosyncratic return conditional on the *lagged magnitude*
of a common 10-session drawdown.  All dates are clipped to the prior completed day.
"""
import numpy as np,pandas as pd,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-02-18'); cal=pd.bdate_range('2020-01-01',END)
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float)
p=pd.DataFrame({a:load(a).reindex(cal).ffill() for a in A});r=p.pct_change();m=r.mean(axis=1)
b=pd.DataFrame({a:r[a].rolling(60,min_periods=42).cov(m)/(m.rolling(60,min_periods=42).var()+1e-12) for a in A});e=r-b.mul(m,axis=0)
# State weight is fully lagged. Larger common drawdowns receive more weight,
# while non-drawdown observations have zero influence.
wealth=(1+m).cumprod(); dd=(1-wealth/wealth.rolling(10,min_periods=8).max()).shift(1).clip(lower=0)
def weighted_mean(x,w,n):
 return x.mul(dd,axis=0).rolling(w,min_periods=n).mean().div(dd.rolling(w,min_periods=n).mean()+1e-12,axis=0)
f=weighted_mean(e,20,14)-weighted_mean(e,60,42)
print('FACTOR severity_weighted_broad_drawdown_residual_resilience_transition_20_60d')
print('VALIDATION_END',END.date(),'CALENDAR_DATES',len(cal),'UNIVERSE',len(A),'DRAW_WEIGHT_FREQUENCY',round(float((dd>0).mean()),6),'MEAN_WEIGHT',round(float(dd.mean()),8))
ics={};summary={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1);out=[];ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z):out.append((t,z));ns.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1)
 summary[h]={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_dates':len(x),'hit_ratio':(x>0).mean(),'mean_valid_instruments':np.mean(ns)}
 print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in summary[h].items()}))
for label,mask in [('2020_2024',ics[10].index<'2025-01-01'),('2025_2026',(ics[10].index>='2025-01-01')&(ics[10].index<'2027-01-01')),('2027_onward',ics[10].index>='2027-01-01')]:
 x=ics[10][mask];print('REGIME_10D',label,'dates',len(x),'ic',round(float(x.mean()),6),'icir',round(float(x.mean()/x.std(ddof=1)),6),'hit',round(float((x>0).mean()),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn))
print('DECAY',json.dumps({str(h):{'ic':round(float(v['daily_paper_ic']),6),'icir':round(float(v['daily_paper_icir']),6),'dates':int(v['ic_dates'])} for h,v in summary.items()}))
print('MAX_ABS_LIBRARY_CORRELATION EVIDENCE_MISSING: full contemporaneous signals for every active admitted factor were not reconstructed; candidate is ineligible for persistence.')

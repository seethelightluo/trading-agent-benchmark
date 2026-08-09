"""One-factor screen: residual downside semivolatility share contraction (60d vs 20d).
Positive signal means recent idiosyncratic downside-risk share has fallen versus its medium-run level.
Completed observations only through 2032-03-03."""
import numpy as np, pandas as pd, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-03-03'); cal=pd.bdate_range('2020-01-01',END)
def get(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float)
p=pd.DataFrame({a:get(a).reindex(cal).ffill() for a in A}); r=p.pct_change(); m=r.mean(axis=1)
# Remove contemporaneous broad market component with trailing 60d beta, then quantify downside share.
b=pd.DataFrame({a:r[a].rolling(60,min_periods=42).cov(m)/(m.rolling(60,min_periods=42).var()+1e-12) for a in A})
e=r-b.mul(m,axis=0)
def downside_share(w,n):
 neg=e.where(e<0).pow(2).rolling(w,min_periods=n).mean()
 tot=e.pow(2).rolling(w,min_periods=n).mean()
 return neg/(tot+1e-12)
f=downside_share(60,42)-downside_share(20,14)
print('FACTOR residual_downside_semivolatility_share_contraction_60_20d')
print('VALIDATION_END',END.date(),'CALENDAR_DATES',len(cal),'UNIVERSE',len(A))
ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1); v=[]; ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z):v.append((t,z));ns.append(len(q))
 x=pd.Series(dict(v),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 print('HORIZON',h,json.dumps({'daily_paper_ic':round(x.mean(),6),'daily_paper_icir':round(x.mean()/sd,6),'ic_standard_error':round(sd/np.sqrt(len(x)),6),'ic_dates':len(x),'hit_ratio':round((x>0).mean(),6),'mean_valid_instruments':round(np.mean(ns),4)}))
for label,mask in [('2020_2024',ics[10].index<'2025-01-01'),('2025_2026',(ics[10].index>='2025-01-01')&(ics[10].index<'2027-01-01')),('2027_onward',ics[10].index>='2027-01-01')]:
 x=ics[10][mask];print('REGIME_10D',label,'dates',len(x),'ic',round(x.mean(),6),'icir',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:ts.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(np.nanmean(ts),6),'TURNOVER_DATES',len(ts))
print('DECAY',json.dumps({str(h):round(x.mean(),6) for h,x in ics.items()}))
print('ORTHOGONALITY_STATUS NOT_TESTED_FULL_LIBRARY: exact contemporaneous implementations unavailable for all admitted signals; any otherwise-qualified candidate cannot be admitted without this binding evidence.')

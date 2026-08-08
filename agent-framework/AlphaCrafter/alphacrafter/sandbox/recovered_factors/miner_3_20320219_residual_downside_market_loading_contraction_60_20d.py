"""One-factor screen: residual continuous downside-market beta contraction (60d minus 20d).
A cross-asset defensive-transition signal: asset residual return sensitivity to broad negative
market days is measured over two windows; positive values indicate reduced recent downside beta.
Uses only completed observations through 2032-02-18."""
import numpy as np, pandas as pd, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-02-18'); cal=pd.bdate_range('2020-01-01',END)
def series(a,col):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,col].astype(float)
p=pd.DataFrame({a:series(a,'close').reindex(cal).ffill() for a in A})
r=p.pct_change(); market=r.mean(axis=1)
# Daily cross-sectional residuals after removing contemporaneous equal-weight market return.
b60=pd.DataFrame({a:r[a].rolling(60,min_periods=42).cov(market)/(market.rolling(60,min_periods=42).var()+1e-12) for a in A})
e=r-b60.mul(market,axis=0)
driver=market.clip(upper=0) # continuous downside driver; zero on non-down days
# rolling covariance beta with adequate nonzero-event history in each window
def beta(x,y,w,n):
 return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/(y.rolling(w,min_periods=n).var()+1e-12) for a in A})
f=beta(e,driver,60,42)-beta(e,driver,20,14)
print('FACTOR residual_continuous_downside_market_loading_contraction_60_20d')
print('VALIDATION_END',END.date(),'CALENDAR_DATES',len(cal),'UNIVERSE',len(A),'DOWNSIDE_DRIVER_FREQUENCY',round(float((driver<0).mean()),6))
allics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1); vals=[]; counts=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'factor':f.loc[t],'forward':fw.loc[t]}).dropna()
  if len(q)>=8 and q.factor.nunique()>1:
   z=q.factor.corr(q.forward,method='spearman')
   if pd.notna(z): vals.append((t,z));counts.append(len(q))
 x=pd.Series(dict(vals),dtype=float); allics[h]=x; sd=x.std(ddof=1)
 print('HORIZON',h,json.dumps({'daily_paper_ic':round(float(x.mean()),6),'daily_paper_icir':round(float(x.mean()/sd),6),'ic_standard_error':round(float(sd/np.sqrt(len(x))),6),'ic_dates':len(x),'hit_ratio':round(float((x>0).mean()),6),'mean_valid_instruments':round(float(np.mean(counts)),4)}))
for label,mask in [('2020_2024',allics[10].index<'2025-01-01'),('2025_2026',(allics[10].index>='2025-01-01')&(allics[10].index<'2027-01-01')),('2027_onward',allics[10].index>='2027-01-01')]:
 x=allics[10][mask]; print('REGIME_10D',label,'dates',len(x),'ic',round(float(x.mean()),6),'icir',round(float(x.mean()/x.std(ddof=1)),6) if len(x)>1 else None,'hit',round(float((x>0).mean()),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(turns)),6),'TURNOVER_DATES',len(turns))
print('DECAY',json.dumps({str(h):round(float(x.mean()),6) for h,x in allics.items()}))
print('ORTHOGONALITY_STATUS NOT_TESTED_FULL_LIBRARY: exact reconstructable contemporaneous signals are unavailable for all 30 admitted factors; binding gate cannot be satisfied.')

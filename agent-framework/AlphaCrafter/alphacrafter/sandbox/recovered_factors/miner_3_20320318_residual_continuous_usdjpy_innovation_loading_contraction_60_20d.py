"""One-factor screen: residual continuous USDJPY innovation loading contraction (60d vs 20d).
Positive values identify assets whose idiosyncratic return sensitivity to continuously
observed USDJPY innovations has declined recently. Data are completed through 2032-03-17."""
import numpy as np, pandas as pd, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-03-17'); cal=pd.bdate_range('2020-01-01',END)
def close(path):
 return pd.read_csv(path,parse_dates=['date']).set_index('date').sort_index()['close'].astype(float)
p=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv').reindex(cal).ffill() for a in A})
r=p.pct_change(); market=r.mean(axis=1)
# First remove the contemporaneous equal-weight cross-asset component.
beta=pd.DataFrame({a:r[a].rolling(60,min_periods=42).cov(market)/(market.rolling(60,min_periods=42).var()+1e-12) for a in A})
e=r-beta.mul(market,axis=0)
fx=close('../persistent/index_data/USDJPY.csv').reindex(cal).ffill().pct_change()
# Innovations are demeaned only using preceding information (60d rolling mean).
innov=fx-fx.rolling(60,min_periods=42).mean()
def loading(w,n):
 return pd.DataFrame({a:e[a].rolling(w,min_periods=n).cov(innov)/(innov.rolling(w,min_periods=n).var()+1e-12) for a in A})
f=loading(60,42)-loading(20,14)
print('FACTOR residual_continuous_usdjpy_innovation_loading_contraction_60_20d')
print('VALIDATION_END',END.date(),'CALENDAR_DATES',len(cal),'UNIVERSE',len(A))
ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1); vals=[]; ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z): vals.append((t,z));ns.append(len(q))
 x=pd.Series(dict(vals),dtype=float); ics[h]=x; sd=x.std(ddof=1)
 print('HORIZON',h,json.dumps({'daily_paper_ic':round(x.mean(),6),'daily_paper_icir':round(x.mean()/sd,6),'ic_standard_error':round(sd/np.sqrt(len(x)),6),'ic_dates':len(x),'hit_ratio':round((x>0).mean(),6),'mean_valid_instruments':round(float(np.mean(ns)),4)}))
for name,mask in [('2020_2024',ics[10].index<'2025-01-01'),('2025_2026',(ics[10].index>='2025-01-01')&(ics[10].index<'2027-01-01')),('2027_onward',ics[10].index>='2027-01-01')]:
 x=ics[10][mask]; print('REGIME_10D',name,'dates',len(x),'ic',round(x.mean(),6),'icir',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn))
print('DECAY',json.dumps({str(h):round(float(x.mean()),6) for h,x in ics.items()}))
print('ORTHOGONALITY_STATUS NOT_TESTED_FULL_LIBRARY: exact contemporaneous signal reconstruction remains unavailable for all admitted factors; missing required full-library correlation evidence prohibits admission.')

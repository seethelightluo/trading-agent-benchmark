"""One-factor screen: residualized short-horizon reversal scaled by idiosyncratic risk."""
import numpy as np, pandas as pd, json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-06-23'); cal=pd.bdate_range('2020-01-01',END)
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float)
p=pd.DataFrame({a:load(a).reindex(cal).ffill() for a in A}); r=p.pct_change(); m=r.mean(axis=1)
# Strip a rolling broad-market beta, then penalize a large recent idiosyncratic move.
# Scaling by residual volatility makes scores comparable among equity, commodity, crypto and yield series.
b=pd.DataFrame({a:r[a].rolling(60,min_periods=42).cov(m)/(m.rolling(60,min_periods=42).var()+1e-12) for a in A})
e=r-b.mul(m,axis=0)
f=-e.rolling(5,min_periods=4).sum().div(e.rolling(20,min_periods=14).std()+1e-12)
print('FACTOR residual_volatility_scaled_reversal_5_20d')
print('VALIDATION_END',END.date(),'CALENDAR_DATES',len(cal),'UNIVERSE',len(A))
ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1); vals=[]; ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z): vals.append((t,z)); ns.append(len(q))
 x=pd.Series(dict(vals),dtype=float); ics[h]=x; sd=x.std(ddof=1)
 print('HORIZON',h,json.dumps({'daily_paper_ic':round(x.mean(),6),'daily_paper_icir':round(x.mean()/sd,6),'ic_standard_error':round(sd/np.sqrt(len(x)),6),'ic_dates':len(x),'hit_ratio':round((x>0).mean(),6),'mean_valid_instruments':round(float(np.mean(ns)),4)}))
for name,mask in [('2020_2024',ics[5].index<pd.Timestamp('2025-01-01')),('2025_2026',(ics[5].index>=pd.Timestamp('2025-01-01'))&(ics[5].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[5].index>=pd.Timestamp('2027-01-01'))]:
 x=ics[5][mask]; print('REGIME_5D',name,'dates',len(x),'ic',round(x.mean(),6),'icir',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn))
print('DECAY',json.dumps({str(h):round(float(x.mean()),6) for h,x in ics.items()}))
print('ORTHOGONALITY_STATUS PENDING: exact contemporaneous correlation versus every admitted library factor is required if IC gates pass; absent evidence fails admission.')

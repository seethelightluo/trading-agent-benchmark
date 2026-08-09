"""One-factor validation: VIX-elevated downside market-beta contraction (60d vs 20d)."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-07-07'); cal=pd.bdate_range('2020-01-01',END)
def close_asset(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float)
p=pd.DataFrame({a:close_asset(a).reindex(cal).ffill() for a in A}); r=p.pct_change(); m=r.mean(axis=1)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].astype(float).reindex(cal).ffill()
vz=(v-v.rolling(60,min_periods=40).mean())/(v.rolling(60,min_periods=40).std()+1e-12)
# Only days simultaneously in a negative broad-market session and above-normal VIX are used.
state=(m<0)&(vz>0)
def conditional_beta(w,minp):
 d=m.where(state); den=d.rolling(w,min_periods=minp).var()+1e-12
 return pd.DataFrame({a:r[a].where(state).rolling(w,min_periods=minp).cov(d)/den for a in A})
# Lower recent stress beta than structural stress beta is the signal (contraction).
f=conditional_beta(60,18)-conditional_beta(20,7)
print('FACTOR vix_elevated_downside_market_beta_contraction_60_20d')
print('VALIDATION_END',END.date(),'PANEL',p.index.min().date(),p.index.max().date(),'UNIVERSE',len(A),'STATE_FRACTION',round(float(state.mean()),6))
ics={}
for h in [1,5,10,20]:
 fw=p.shift(-h).div(p).sub(1); rows=[]; ns=[]
 for t in f.index[:-h]:
  q=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=q.f.corr(q.y,method='spearman')
   if pd.notna(z): rows.append((t,z));ns.append(len(q))
 x=pd.Series(dict(rows),dtype=float);ics[h]=x; sd=x.std(ddof=1)
 print('HORIZON',h,json.dumps({'daily_paper_ic':round(float(x.mean()),6),'daily_paper_icir':round(float(x.mean()/sd),6),'ic_standard_error':round(float(sd/np.sqrt(len(x))),6),'ic_dates':len(x),'hit_ratio':round(float((x>0).mean()),6),'mean_valid_instruments':round(float(np.mean(ns)),3)}))
for n,mask in [('2020_2024',ics[10].index<pd.Timestamp('2025-01-01')),('2025_2026',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
 x=ics[10][mask]; print('REGIME_10D',n,'dates',len(x),'ic',round(float(x.mean()),6),'icir',round(float(x.mean()/x.std(ddof=1)),6),'hit',round(float((x>0).mean()),6))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn))
print('DECAY',json.dumps({str(h):round(float(x.mean()),6) for h,x in ics.items()}))
print('ORTHOGONALITY_STATUS PENDING: only reconstruct against every admitted factor if IC/ICIR gates pass.')

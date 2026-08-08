"""miner_2 revalidation of admitted realized-volatility factor through prior completed day."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-08-26')
P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 P[a]=d.close.astype(float)
p=pd.DataFrame(P).sort_index(); r=p.pct_change(fill_method=None)
f=r.rolling(20,min_periods=15).std()
def icstats(h):
 fw=pd.DataFrame({a:P[a].shift(-h)/P[a]-1 for a in A})
 vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d].rename('factor'),fw.loc[d].rename('forward')],axis=1).dropna()
  if len(z)>=8:
   vals.append((d,spearmanr(z.factor,z.forward).statistic));ns.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_std':float(sd),'ic_hit_ratio':float((s>0).mean()),'ic_dates':len(s),'mean_valid_instruments_per_ic_date':float(np.mean(ns))}
res={}
for h in [1,5,10,20]:
 s,m=icstats(h);res[h]=(s,m);print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for n,mask in [('2020',s.index<'2021-01-01'),('2021_2022',(s.index>='2021-01-01')&(s.index<'2023-01-01')),('2023_2024',(s.index>='2023-01-01')&(s.index<'2025-01-01')),('2025_2026_08_26',s.index>='2025-01-01'),('latest_90_calendar_days',s.index>=END-pd.Timedelta(days=90))]:
   q=s[mask];print('REGIME',n,'dates',len(q),'ic',float(q.mean()),'icir',float(q.mean()/q.std(ddof=1)),'hit',float((q>0).mean()))
ranks=f.rank(axis=1,pct=True); turn=(ranks-ranks.shift(10)).abs().stack().mean()
print('FACTOR miner_2_realized_volatility_20obs')
print('VALIDATION_DATE',END.date(),'signal_start',f.index.min().date(),'signal_dates',len(f),'universe',len(A),'coverage',float(f.notna().sum().sum()/f.size),'mean_names_per_signal_date',float(f.notna().sum(axis=1).mean()),'turnover_10d_rank',float(turn))
print('DECAY',json.dumps({str(h):m for h,(s,m) in res.items()},sort_keys=True))
print('ADMISSION_PRIMARY_10D',json.dumps(res[10][1],sort_keys=True))

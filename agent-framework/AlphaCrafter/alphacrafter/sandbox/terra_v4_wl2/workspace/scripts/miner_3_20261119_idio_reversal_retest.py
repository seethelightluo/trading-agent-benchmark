import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15'); rows_by={}
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U}
# Idiosyncratic reversal: negative trailing residual return vs cross-asset median on that asset's own sessions.
for w in [2,3,5,10]:
 rows=[]
 for s,p in D.items():
  r=p.pct_change(); med=pd.DataFrame({k:v.pct_change() for k,v in D.items()}).median(axis=1).reindex(r.index); resid=r-med
  f=-resid.rolling(w,min_periods=w).sum(); y=p.shift(-1)/p-1
  rows += [(dt,s,float(f.loc[dt]),float(y.loc[dt])) for dt in p.index if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt])]
 a=pd.DataFrame(rows,columns=['date','s','f','y']); z=[]; ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:z.append(spearmanr(g.f,g.y).statistic);ns.append(len(g))
 z=np.array(z); ranks=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank');
 print(f'w {w} dates {len(z)} avg_names {np.mean(ns):.2f} coverage {a.s.nunique()/15:.2%} IC {z.mean():.6f} ICIR {z.mean()/z.std(ddof=1):.6f} hit {np.mean(z>0):.4f} turnover {ranks.diff().abs().mean().mean():.4f}')
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=z[[d.year>=lo and d.year<=hi for d in sorted(a.date.unique())[:0]]] if False else []
  dates=[dt for dt,g in a.groupby('date') if lo<=dt.year<=hi and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1]
  vals=[spearmanr(a[a.date==dt].f,a[a.date==dt].y).statistic for dt in dates]; print(' regime',lo,hi,len(vals),np.mean(vals),np.mean(vals)/np.std(vals,ddof=1))
 # horizons
 for h in [5,10]:
  yh=[]
  for s,p in D.items():
   r=p.pct_change(); med=pd.DataFrame({k:v.pct_change() for k,v in D.items()}).median(axis=1).reindex(r.index); f=- (r-med).rolling(w,min_periods=w).sum(); y=p.shift(-h)/p-1
   yh += [(dt,float(f.loc[dt]),float(y.loc[dt])) for dt in p.index if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt])]
  aa=pd.DataFrame(yh,columns=['date','f','y']); zz=[spearmanr(g.f,g.y).statistic for _,g in aa.groupby('date') if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1]; print(' horizon',h,'IC',np.mean(zz),'ICIR',np.mean(zz)/np.std(zz,ddof=1),'dates',len(zz))

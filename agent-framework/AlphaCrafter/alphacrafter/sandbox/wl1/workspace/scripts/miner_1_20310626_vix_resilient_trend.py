import pandas as pd, numpy as np, os
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data')
px={}
for s in U:
 d=pd.read_csv(base/(s+'.csv'))
 d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
P=P.loc[:'2031-06-26']
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date']); vix=vix.sort_values('date').set_index('date')['close'].astype(float).reindex(P.index).ffill()
# lagged signal: 20d vol-adjusted trend, with resilience bonus based on return on high-VIX days
R=P.pct_change()
vix_hi=vix>vix.rolling(120,min_periods=60).quantile(.7)
res={}
for s in U:
 r=R[s]
 trend=P[s].shift(1).pct_change(20)
 vol=r.shift(1).rolling(20,min_periods=15).std()*np.sqrt(20)
 high=(r.shift(1)*vix_hi.shift(1)).rolling(60,min_periods=30).mean()
 low=(r.shift(1)*(~vix_hi.shift(1).fillna(False)).astype(float)).rolling(60,min_periods=30).mean()
 # high-vol resilience relative to low-vol behavior, scaled modestly
 res[s]=trend/vol + 0.5*(high-low)/r.shift(1).rolling(60,min_periods=30).std()
F=pd.DataFrame(res)
# cross-sectional rank; forward daily returns (strictly future)
rows=[]; sigvals=[]
for dt in F.index:
 f=F.loc[dt]; fw=R.shift(-1).loc[dt]
 ok=f.notna()&fw.notna()
 if ok.sum()>=8:
  x=f[ok].rank(pct=True); y=fw[ok]
  ic=x.corr(y)
  rows.append((dt,ic,ok.sum()))
  sigvals.append(x)
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
def stat(z): return (z.mean(), z.mean()/z.std(ddof=1), (z>0).mean(),len(z))
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
print('daily mean IC, ICIR, hit, n',stat(q.ic))
for k in [5,10,20]:
 fwd=P.shift(-k)/P-1; rr=[]
 for dt in F.index:
  a=F.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: rr.append(a[ok].rank(pct=True).corr(b[ok]))
 print('decay',k,stat(pd.Series(rr).dropna()))
# turnover using rank vectors
rank=F.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna()
print('turnover',turn.mean())
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-12-31'),('2029','2030-12-31'),('2031','2031-12-31')]:
 z=q.loc[a:b,'ic']; print('regime',a,b,stat(z) if len(z)>2 else None)
print('period',q.index.min().date(),q.index.max().date())
# artifact
F.to_csv('scripts/miner_1_20310626_vix_resilient_trend_signal.csv',index_label='date')

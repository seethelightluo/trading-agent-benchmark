import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-07-15'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); D[s]=x[x.index<=cut]
# Downside-risk asymmetry: prefer assets with lower realized downside semideviation,
# while retaining enough upside activity. Signal = - downside semideviation over trailing 20 sessions.
rows=[]
for s,x in D.items():
 r=x.close.pct_change(); down=r.where(r<0)
 f=-down.rolling(20,min_periods=12).std()
 for dt in x.index:
  if pd.notna(f.get(dt)) and dt in x.index:
   y=x.close.shift(-1)/x.close-1
   if pd.notna(y.loc[dt]): rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
a=pd.DataFrame(rows,columns=['date','symbol','factor','forward']); z=[]; ns=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1:z.append(spearmanr(g.factor,g.forward).statistic);ns.append(len(g))
z=np.array(z);print('dates',len(z),'avg_names',np.mean(ns),'coverage',a.symbol.nunique()/15);print('daily IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',a.assign(rank=a.groupby('date').factor.rank(pct=True)).pivot(index='date',columns='symbol',values='rank').diff().abs().mean().mean())
for h in [5,10]:
 q=[]
 for s,x in D.items():
  f=-x.close.pct_change().where(x.close.pct_change()<0).rolling(20,min_periods=12).std(); y=x.close.shift(-h)/x.close-1
  for dt in x.index:
   if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]):q.append((dt,s,f.loc[dt],y.loc[dt]))
 q=pd.DataFrame(q,columns=['date','s','f','y']);v=[spearmanr(g.f,g.y).statistic for _,g in q.groupby('date') if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1];v=np.array(v);print(h,'d IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'dates',len(v))
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-07-15')]:
 v=[spearmanr(g.factor,g.forward).statistic for dt,g in a.groupby('date') if lo<=str(dt.date())<=hi and len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1];v=np.array(v);print(label,len(v),v.mean(),v.mean()/v.std(ddof=1))

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-10-07')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut] for s in U}
# Volatility-scaled session spread: reversal of overnight plus intraday return, scaled by prior 20d daily volatility.
rows=[]
for s,x in D.items():
 gap=x.open/x.close.shift(1)-1; intra=x.close/x.open-1; vol=x.close.pct_change().rolling(20,min_periods=15).std().shift(1)
 f=(-gap-intra)/vol; y=x.close.shift(-1)/x.close-1
 for dt in x.index:
  if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): rows.append((dt,s,f.loc[dt],y.loc[dt]))
a=pd.DataFrame(rows,columns=['date','s','f','y'])
def calc(df,ycol='y'):
 z=[]; ns=[]
 for dt,g in df.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g[ycol].nunique()>1: z.append(spearmanr(g.f,g[ycol]).statistic);ns.append(len(g))
 z=np.asarray(z); return len(z),np.mean(ns),z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0),z
n,an,ic,ir,hit,z=calc(a); print(f'dates {n} avg_names {an:.2f} coverage {a.s.nunique()/15:.4f} IC {ic:.6f} ICIR {ir:.6f} hit {hit:.4f}')
for h in [5,10]:
 q=[]
 for s,x in D.items():
  gap=x.open/x.close.shift(1)-1; intra=x.close/x.open-1; vol=x.close.pct_change().rolling(20,min_periods=15).std().shift(1); f=(-gap-intra)/vol; y=x.close.shift(-h)/x.close-1
  q += [(dt,s,f.loc[dt],y.loc[dt]) for dt in x.index if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt])]
 q=pd.DataFrame(q,columns=['date','s','f','y']);n2,an2,ic2,ir2,_,_=calc(q);print(f'h{h} dates {n2} IC {ic2:.6f} ICIR {ir2:.6f}')
r=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank');print('turnover',r.diff().abs().mean().mean())
for lab,lo,hi in [('2020-22',2020,2022),('2023-24',2023,2024),('2025-26',2025,2026)]:
 v=a[(a.date.dt.year>=lo)&(a.date.dt.year<=hi)];nn,_,ii,iiir,_,_=calc(v);print(lab,'dates',nn,'IC',round(ii,6),'ICIR',round(iiir,6))
# Correlation of pooled cross-sectional ranks to existing raw session spread and common signals
base=[]
for s,x in D.items():
 f=-(x.open/x.close.shift(1)-1)-(x.close/x.open-1); base += [(dt,s,f.loc[dt]) for dt in x.index if pd.notna(f.loc[dt])]
b=pd.DataFrame(base,columns=['date','s','b']); m=a.merge(b,on=['date','s']);print('pooled_rank_corr_raw_spread',m.f.corr(m.b,method='spearman'))
print('valid_rows',len(a))

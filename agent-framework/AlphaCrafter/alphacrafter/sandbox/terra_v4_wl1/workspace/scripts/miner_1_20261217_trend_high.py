import pandas as pd,numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=cut]; R=P.pct_change()
# Trend persistence: 20d return, penalized by distance below 60d high; cross-sectional rank blend.
mom20=R.rolling(20,min_periods=20).sum(); high60=P.rolling(60,min_periods=60).max(); draw=(P/high60-1).clip(-1,0)
# favor positive medium trend and assets near highs; robust ranks reduce scale effects
F=mom20.rank(axis=1,pct=True)+0.5*(draw.rank(axis=1,pct=True)); F=F.sub(F.median(axis=1),axis=0)
F.to_csv('scripts/miner_1_20261217_trend_high_signal.csv',index_label='date')
for h in [1,5,10]:
 Y=P.pct_change(h).shift(-h); vals=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z))
 a=pd.Series(vals); print('H',h,'dates',len(a),'avg_names',round(np.mean(ns),2),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1),8),'hit',round((a>0).mean(),4))
print('coverage',round(F.notna().sum().sum()/F.size,6),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
# regime halves
Y=P.pct_change(1).shift(-1); vals=[]
for d in F.index:
 z=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
 if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: vals.append((d,spearmanr(z.f,z.y).statistic))
a=pd.DataFrame(vals,columns=['date','ic']); a['year']=a.date.dt.year
print('yearly',a.groupby('year').ic.agg(['count','mean']).round(5).to_dict('index'))
cs=[]
for p in Path('scripts').glob('*signal.csv'):
 try:
  x=pd.read_csv(p,index_col=0,parse_dates=True).reindex(F.index).reindex(columns=U); c=F.stack().corr(x.stack())
  if pd.notna(c): cs.append((abs(c),p.name,c))
 except: pass
print('max_abs_library_correlation',max(cs)[0] if cs else None, sorted(cs,reverse=True)[:5]); print('period',F.index.min().date(),F.index.max().date())

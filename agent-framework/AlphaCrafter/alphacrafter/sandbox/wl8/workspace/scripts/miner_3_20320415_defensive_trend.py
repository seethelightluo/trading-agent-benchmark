import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-04-15')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
r=p.pct_change()
# Defensive trend: intermediate trend rewarded only when its own risk is low;
# cross-asset stress gate shifts signal toward low-volatility assets.
ret20=r.rolling(20,min_periods=20).sum().shift(1)
vol30=r.rolling(30,min_periods=30).std().shift(1)
cs_stress=vol30.median(axis=1).rolling(10,min_periods=10).mean()
stress_cut=cs_stress.rolling(252,min_periods=100).quantile(.70).shift(1)
gate=(1.0-0.45*(cs_stress>stress_cut).astype(float))
f=(ret20/(vol30*np.sqrt(20)+1e-12))*gate.values[:,None]
# add a modest low-risk preference in stressed conditions
f=f-0.35*(vol30.rank(axis=1,pct=True).sub(.5)).mul((cs_stress>stress_cut).astype(float),axis=0)
fr=p.shift(-10)/p-1
rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 a=f.loc[d]; b=fr.loc[d]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3: rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; q=[]
 for d in x.index:
  a=f.loc[d]; b=yy.loc[d]; ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2030',z['2030']),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
f.loc[x.index].to_csv('scripts/miner_3_20320415_defensive_trend_signal.csv'); x.to_csv('scripts/miner_3_20320415_defensive_trend_ic.csv')

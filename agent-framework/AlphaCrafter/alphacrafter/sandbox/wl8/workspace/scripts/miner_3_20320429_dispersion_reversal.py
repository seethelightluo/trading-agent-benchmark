import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-04-15')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
r=p.pct_change()
ret5=r.rolling(5,min_periods=5).sum().shift(1)
vol20=r.rolling(20,min_periods=20).std().shift(1)
disp=r.rolling(10,min_periods=10).std().mean(axis=1).shift(1)
threshold=disp.rolling(120,min_periods=60).quantile(.65).shift(1)
gate=(disp>threshold).astype(float)
f=(-ret5/(vol20*np.sqrt(5)+1e-12)).mul(gate,axis=0)
f=f.rolling(2,min_periods=2).mean().shift(1)
fr=p.shift(-10)/p-1
rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 a=f.loc[d];b=fr.loc[d];ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3: rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in x.index:
  a=f.loc[d];b=yy.loc[d];ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2030',z['2030']),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
f.loc[x.index].to_csv('scripts/miner_3_20320429_dispersion_reversal_signal.csv');x.to_csv('scripts/miner_3_20320429_dispersion_reversal_ic.csv')

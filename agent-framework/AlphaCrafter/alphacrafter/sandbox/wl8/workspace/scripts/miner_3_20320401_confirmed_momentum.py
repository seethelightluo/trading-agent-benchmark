import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-03-18')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
# Volatility-adjusted intermediate momentum, confirmed by positive short-term return.
# All rolling inputs are shifted so signal at d uses data through d-1.
mom=r.rolling(20,min_periods=20).sum().shift(1); confirm=r.rolling(5,min_periods=5).sum().shift(1); vol=r.rolling(30,min_periods=30).std().shift(1)
f=(mom/(vol*np.sqrt(20)+1e-12))*np.where(confirm>0,1.0,0.35)
fr=p.shift(-10)/p-1; rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 a=f.loc[d]; b=fr.loc[d]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3: rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in x.index:
  a=f.loc[d];b=yy.loc[d];ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2030',z['2030']),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
f.loc[x.index].to_csv('scripts/miner_3_20320401_confirmed_momentum_signal.csv');x.to_csv('scripts/miner_3_20320401_confirmed_momentum_ic.csv')

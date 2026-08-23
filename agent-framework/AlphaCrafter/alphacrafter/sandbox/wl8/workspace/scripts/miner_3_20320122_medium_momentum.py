import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-01-22')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change()
# lag-safe medium momentum, volatility adjusted and beta-neutral to equal-weight market
m=r.rolling(60,min_periods=60).sum().shift(1); v=r.rolling(40,min_periods=40).std().shift(1); f=m/(v+1e-12)
market=r.mean(axis=1); beta=r.rolling(60).cov(market).div(market.rolling(60).var(),axis=0).shift(1); f=f-beta.mul(market.rolling(60).sum().shift(1),axis=0)*0.15
fr=p.shift(-10)/p-1; rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 a=f.loc[d]; b=fr.loc[d]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3: rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;q=[]
 for d in x.index:
  a=f.loc[d];b=yy.loc[d];ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2020',z['2020']),('2021',z['2021']),('2022',z['2022']),('2023',z['2023']),('2024',z['2024']),('2025',z['2025']),('2026',z['2026']),('2027',z['2027']),('2028',z['2028']),('2029',z['2029']),('2030',z['2030']),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
pd.DataFrame({s:f.loc[x.index,s] for s in U}).to_csv('scripts/miner_3_20320122_medium_momentum_signal.csv');x.to_csv('scripts/miner_3_20320122_medium_momentum_ic.csv')

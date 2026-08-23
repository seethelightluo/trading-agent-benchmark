import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
r=p.pct_change(); defs=['XAU','US10Y','CN10Y']; dr=r[defs].mean(axis=1)
# Lagged 30-day excess trend against defensive basket, scaled by prior 45-day volatility.
f=(r.rolling(30,min_periods=30).sum().sub(dr.rolling(30,min_periods=30).sum(),axis=0).div(r.rolling(45,min_periods=45).std().shift(1),axis=0)).shift(1)
fr=p.shift(-10)/p-1; rows=[]
for i,d in enumerate(p.index[:-10]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>pd.Timestamp('2032-03-04'): continue
 a=f.loc[d]; b=fr.loc[d]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>=3: rows.append((d,spearmanr(a[ok],b[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; q=[]
 for d in x.index:
  a=f.loc[d]; b=yy.loc[d]; ok=a.notna()&b.notna()
  if ok.sum()>=8:q.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('60',z.tail(60)),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
f.to_csv('scripts/miner_2_20320304_defensive_relative_trend30_signal.csv'); x.to_csv('scripts/miner_2_20320304_defensive_relative_trend30_ic.csv')

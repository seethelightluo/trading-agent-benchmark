import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-07-08')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill(); r=p.pct_change(); defs=['XAU','US10Y','CN10Y']
# Slow relative trend: 60d asset return relative to defensive basket, normalized by each asset's trailing volatility.
a=p.shift(1)/p.shift(61)-1; d=a[defs].mean(axis=1); vol=r.shift(1).rolling(40,min_periods=30).std()*np.sqrt(252); f=a.sub(d,axis=0).div(vol).rolling(3,min_periods=3).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
fr={h:p.shift(-h)/p-1 for h in [1,5,10,20]}; rows=[]
for i,dt in enumerate(p.index[:-20]):
 if dt<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 q=ic(f.loc[dt],fr[10].loc[dt]);
 if pd.notna(q):rows.append((dt,q,(f.loc[dt].notna()&fr[10].loc[dt].notna()).sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z=x.ic
print('dates',len(z),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=[ic(f.loc[dt],fr[h].loc[dt]) for dt in x.index];q=[v for v in q if pd.notna(v)];print('decay',h,np.mean(q),len(q))
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2032',z[z.index.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_2_20320805_slow_relative_trend_signal.csv');x.to_csv('scripts/miner_2_20320805_slow_relative_trend_ic.csv')

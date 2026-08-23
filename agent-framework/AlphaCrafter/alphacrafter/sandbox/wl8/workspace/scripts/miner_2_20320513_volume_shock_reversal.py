import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-05-13')
P={}; V={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 P[s]=d.close; V[s]=d.volume
p=pd.DataFrame(P).ffill(); v=pd.DataFrame(V).reindex(p.index).ffill(); r=p.pct_change()
# Volume-shock confirmation: fade recent 5d cross-sectional moves, with larger reversal
# exposure when lagged volume is unusually high; all inputs lagged before forward return.
ret5=r.rolling(5,min_periods=5).sum().shift(1)
vol20=r.rolling(20,min_periods=20).std().shift(1)
vs=(v/(v.rolling(20,min_periods=10).median().shift(1)+1e-12)).clip(0.25,4)
# cross-sectional volume surprise, capped to avoid crypto/market-volume scale artifacts
shock=np.log(vs).clip(-1,1)
f=(-ret5/(vol20*np.sqrt(5)+1e-12))*shock
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
for n,q in [('365',z.tail(365)),('180',z.tail(180)),('2031',z['2031']),('2032',z['2032'])]: print(n,q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
f.loc[x.index].to_csv('scripts/miner_2_20320513_volume_shock_reversal_signal.csv'); x.to_csv('scripts/miner_2_20320513_volume_shock_reversal_ic.csv')

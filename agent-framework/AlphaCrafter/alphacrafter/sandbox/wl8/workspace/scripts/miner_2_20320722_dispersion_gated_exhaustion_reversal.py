import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-07-08')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill()
r=p.pct_change(); csdisp=r.rolling(20,min_periods=15).std().mean(axis=1)
# Reversal is strongest after an unusually dispersed cross-asset shock; scale by own volatility.
shock=p.shift(1)/p.shift(6)-1
vol=r.shift(1).rolling(20,min_periods=15).std()
z=shock.div(vol).replace([np.inf,-np.inf],np.nan)
gate=(csdisp.shift(1)>csdisp.shift(1).rolling(120,min_periods=60).median()).astype(float)
f=(-z.mul(gate,axis=0)).rolling(3,min_periods=3).mean()
def ic(a,b):
 ok=a.notna()&b.notna()
 if ok.sum()<8 or a[ok].nunique()<3 or b[ok].nunique()<3:return np.nan
 return spearmanr(a[ok],b[ok]).statistic
fr={h:p.shift(-h)/p-1 for h in [1,5,10,20]}; rows=[]
for i,d in enumerate(p.index[:-20]):
 if d<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 q=ic(f.loc[d],fr[10].loc[d])
 if pd.notna(q):rows.append((d,q,(f.loc[d].notna()&fr[10].loc[d].notna()).sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); zic=x.ic
print('dates',len(zic),'start',x.index.min().date(),'end',x.index.max().date(),'avg_n',x.n.mean(),'coverage',f.loc[x.index].notna().mean().mean(),'IC',zic.mean(),'ICIR',zic.mean()/zic.std(ddof=1),'hit',(zic>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean().mean())
for h in [1,5,10,20]:
 q=[ic(f.loc[d],fr[h].loc[d]) for d in x.index];q=[v for v in q if pd.notna(v)];print('decay',h,np.mean(q),len(q))
for n,q in [('365',zic.tail(365)),('180',zic.tail(180)),('2032',zic[zic.index.year==2032])]:print(n,q.mean(),q.mean()/q.std(ddof=1),len(q))
f.loc[x.index].to_csv('scripts/miner_2_20320722_exhaustion_signal.csv');x.to_csv('scripts/miner_2_20320722_exhaustion_ic.csv')

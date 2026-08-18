import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().ffill(); L=np.log(P); R=L.diff()
rel=L.diff(60).sub(L.diff(60).median(axis=1),axis=0)
agree=(np.sign(L.diff(10))+np.sign(L.diff(30))+np.sign(L.diff(60)))/3
# Cross-sectional percentile transform prevents crypto/commodity scale dominance; agreement supplies persistence.
f=rel.rank(axis=1,pct=True).sub(.5).mul(agree).shift(1)
y=L.shift(-10)-L; rows=[]
for dt in f.index:
 a,b=f.loc[dt],y.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
z.to_csv('scripts/miner_2_20330930_rank_agreement60_ic.csv'); f.to_csv('scripts/miner_2_20330930_rank_agreement60_signal.csv')

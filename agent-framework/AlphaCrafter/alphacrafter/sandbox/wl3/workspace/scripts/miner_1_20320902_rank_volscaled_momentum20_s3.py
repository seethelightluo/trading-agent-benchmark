import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().ffill();R=np.log(P).diff()
raw=P.pct_change(20)/(R.rolling(40,min_periods=25).std()*np.sqrt(252));f=raw.rank(axis=1,pct=True).rolling(3,min_periods=3).mean().shift(1);fr=np.log(P.shift(-10)/P);rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8:rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n);print('recent',n,x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_1_20320902_rank_volscaled_momentum20_s3_signal.csv');z.to_csv('scripts/miner_1_20320902_rank_volscaled_momentum20_s3_ic.csv')

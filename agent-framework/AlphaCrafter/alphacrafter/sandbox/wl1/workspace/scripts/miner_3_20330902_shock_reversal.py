import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 q=pd.read_csv('../persistent/stock_data/'+s+'.csv'); q.date=pd.to_datetime(q.date); D[s]=q.sort_values('date').set_index('date').close.astype(float)
p=pd.concat(D,axis=1).sort_index().ffill(); r=np.log(p).diff(); bench=r.mean(axis=1)
# Shock-reversal only when cross-asset dispersion is elevated; magnitude scaled by recent volatility.
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
th=disp.rolling(120,min_periods=60).quantile(.70)
vol=r.rolling(20,min_periods=15).std()
f=(-np.log(p).diff(5)/(vol*np.sqrt(5)+1e-8)).where(disp>th,0).shift(1)
y=np.log(p).shift(-10)-np.log(p); rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2024-01-01':'2033-08-20']
print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f active %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean(),(f!=0).sum().sum()/f.notna().sum().sum()))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=x.loc[a:b];print(a,b,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(),6))
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).loc[x.index].mean())
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_3_20330902_shock_reversal_signal.csv',index=False)

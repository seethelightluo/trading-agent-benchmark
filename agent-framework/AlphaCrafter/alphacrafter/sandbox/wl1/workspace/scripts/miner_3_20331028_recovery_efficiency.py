import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 q=pd.read_csv('../persistent/stock_data/'+s+'.csv'); q.date=pd.to_datetime(q.date)
 D[s]=q.sort_values('date').set_index('date').close.astype(float)
p=pd.concat(D,axis=1).sort_index().ffill(); lr=np.log(p).diff()
# Recovery efficiency: trailing return relative to the worst peak-to-trough loss,
# with a small floor. Lagged one session. High values favor assets recovering
# efficiently from drawdowns rather than merely having high raw momentum.
ret20=np.log(p).diff(20)
peak=p.rolling(60,min_periods=40).max(); dd=p/peak-1
mdd=(-dd.rolling(40,min_periods=25).min()).clip(lower=0)
vol=lr.rolling(20,min_periods=15).std()
f=(ret20/(mdd+0.02)/(vol*np.sqrt(20)+1e-8)).shift(1)
y=np.log(p).shift(-10)-np.log(p)
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): rows.append((d,ic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').loc['2024-01-01':'2033-10-26']
print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2032','2033')]:
 q=x.loc[a:b];print(a,b,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(),6))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[x.index].mean())
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_3_20331028_recovery_efficiency_signal.csv',index=False)

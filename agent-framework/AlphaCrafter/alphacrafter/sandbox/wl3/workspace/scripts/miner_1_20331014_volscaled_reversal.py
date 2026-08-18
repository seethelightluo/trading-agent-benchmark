import numpy as np,pandas as pd,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
p=pd.concat(px,axis=1).sort_index(); r=p.pct_change()
# Volatility-scaled short-term reversal, lagged: negative 3d return divided by trailing 20d vol.
f=(-p.pct_change(3)/r.rolling(20).std()).shift(1); rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],(p.shift(-10)/p-1).loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std()*np.sqrt(252),'hit',(a.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 q=a.tail(n);print('recent',n,q.ic.mean(),q.ic.mean()/q.ic.std()*np.sqrt(252))
for h in [1,5,10,20]:
 vals=[]; fw=p.shift(-h)/p-1
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 s=pd.Series(vals);print('decay',h,s.mean(),s.mean()/s.std()*np.sqrt(252))
print('period',a.index.min(),a.index.max())
f.to_csv('scripts/miner_1_20331014_volscaled_reversal_signal.csv');a.to_csv('scripts/miner_1_20331014_volscaled_reversal_ic.csv')

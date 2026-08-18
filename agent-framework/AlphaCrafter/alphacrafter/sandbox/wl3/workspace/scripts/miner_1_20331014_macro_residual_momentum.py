import numpy as np, pandas as pd, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in files}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
prices=pd.concat(px,axis=1).sort_index(); rets=prices.pct_change(); vret=vix.pct_change()
asset20=prices.pct_change(20); v20=vix.pct_change(20)
beta=pd.DataFrame(index=prices.index,columns=prices.columns,dtype=float)
for c in prices:
    beta[c]=rets[c].rolling(60,min_periods=40).cov(vret)/vret.rolling(60,min_periods=40).var()
factor=asset20-beta.mul(v20,axis=0)
signal=factor.shift(1); fwd=prices.shift(-10)/prices-1
rows=[]
for d in signal.index:
 z=pd.concat([signal.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std()*np.sqrt(252),(r.ic>0).mean()))
for n in [120,252,756,1260]:
 q=r.tail(n);print('recent',n,'ICIR',q.ic.mean()/q.ic.std()*np.sqrt(252),'IC',q.ic.mean(),'dates',len(q))
for h in [1,5,10,20]:
 fw=prices.shift(-h)/prices-1; vals=[]
 for d in signal.index:
  z=pd.concat([signal.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 vals=pd.Series(vals);print('decay',h,'IC',vals.mean(),'ICIR',vals.mean()/vals.std()*np.sqrt(252))
print('turnover',signal.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('period',r.index.min(),r.index.max())
signal.to_csv('scripts/miner_1_20331014_macro_residual_momentum_signal.csv')
pd.DataFrame({'date':r.index,'ic':r.ic,'n':r.n}).to_csv('scripts/miner_1_20331014_macro_residual_momentum_ic.csv',index=False)

import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in watch:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); P[s]=x.set_index('date').close
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); m=r.mean(axis=1)
# residual cumulative momentum after removing daily common movement; condition on breadth trend
res=r.sub(m,axis=0)
raw=res.rolling(20).sum(); vol=res.rolling(40).std().replace(0,np.nan); base=raw/vol
bread=(r>0).sum(axis=1)/r.notna().sum(axis=1)
condition=(bread.rolling(10).mean()-0.5)
f=base.mul(condition,axis=0).shift(1); fw=px.shift(-10)/px-1
rows=[]
for d in px.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),(o.ic>0).mean()))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for n in [260,520,780]:
 q=o.tail(n);print('recent',n,'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
for h in [1,5,10,20,30]:
 fw=px.shift(-h)/px-1; x=[]
 for d in px.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'%.8f'%np.nanmean(x),'n',len(x))
sig=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();sig.to_csv('scripts/artifacts/miner_1_20330915_breadth_conditioned_residual_momentum_signal.csv',index=False)

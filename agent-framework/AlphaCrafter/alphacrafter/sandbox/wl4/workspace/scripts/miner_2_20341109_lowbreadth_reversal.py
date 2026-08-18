import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is not None:fs[s]=d[['date','close']].drop_duplicates('date').set_index('date')['close']
p=pd.DataFrame(fs).sort_index();r=p.pct_change();m=r.rolling(20).sum(); breadth=(m>0).sum(1)/m.notna().sum(1)
# In broad weakness use contrarian 20d reversal; neutral otherwise.
f=(-m).shift(1).where(breadth.shift(1)<.4); fw=p.shift(-10)/p-1; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z),breadth.loc[dt]))
o=pd.DataFrame(rows,columns=['date','ic','n','breadth']).set_index('date');print('dates',len(o),'avg_n',o.n.mean(),'active',len(o)/len(f))
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean(),f.notna().sum().sum()/(len(f)*len(U)),f.rank(axis=1).diff().abs().sum().sum()/(len(f)*len(U))))
for n in [120,260,520,780]:
 q=o.tail(n).ic;print('recent',n,q.mean(),q.mean()/q.std(),(q>0).mean(),len(q))
o.to_csv('scripts/artifacts/miner_2_20341109_lowbreadth_reversal_ic.csv');f.to_csv('scripts/artifacts/miner_2_20341109_lowbreadth_reversal_signal.csv')

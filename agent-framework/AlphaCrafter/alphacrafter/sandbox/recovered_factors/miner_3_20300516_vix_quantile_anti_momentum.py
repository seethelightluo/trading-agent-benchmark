import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
px={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')};p=pd.DataFrame(px).sort_index().astype(float)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill(); base=-(p/p.shift(20)-1)
for q in [.6,.7,.8]:
 gate=v>v.rolling(120,min_periods=60).quantile(q); sig=base.where(gate);print('Q',q,'gate',round(gate.mean(),3))
 for h in [1,3,5,10,20]:
  f=p.shift(-h)/p-1;a=[]
  for d in p.index:
   z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.array(a);print(h,len(a),round(a.mean(),5),round(a.mean()/a.std(ddof=1),5))

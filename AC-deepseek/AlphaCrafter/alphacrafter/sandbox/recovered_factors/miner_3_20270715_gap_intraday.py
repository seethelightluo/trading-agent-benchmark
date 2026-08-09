import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}; A=sorted(D)
O=pd.concat({a:D[a]['open'] for a in A},axis=1).sort_index(); C=pd.concat({a:D[a]['close'] for a in A},axis=1).sort_index()
gap=O/C.shift(1)-1; intr=C/O-1
f=(gap*intr).shift(1)
for h in [1,5,10,20]:
 y=C.pct_change(h).shift(-h); vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
y=C.pct_change().shift(-1); vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
z=pd.DataFrame(vals,columns=['date','ic']).set_index('date');print('year',z.groupby(z.index.year).ic.mean().round(5).to_dict());print('coverage',f.notna().sum().sum()/f.size,'valid_avg',f.notna().sum(axis=1).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().sum(axis=1).div(len(A)).dropna().mean())
for w in [60,120,250]:
 q=z.ic.tail(w);print('recent',w,'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))

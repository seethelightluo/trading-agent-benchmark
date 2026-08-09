import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}; fw={1:{},5:{}}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change(); v=r.rolling(20,min_periods=12).std()
 z=(r/v).clip(-6,6); F[a]=-(0.5*z+0.3*z.shift(1)+0.2*z.shift(2))
 for h in fw: fw[h][a]=d.close.pct_change(h).shift(-h)
fac=pd.DataFrame(F); print('assets',len(assets),'rows',len(fac))
for h in fw:
 fwd=pd.DataFrame(fw[h]); vals=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.nunique().min()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 s=pd.Series(vals); print(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean())
print('coverage',fac.notna().sum(axis=1).mean()/15,'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())

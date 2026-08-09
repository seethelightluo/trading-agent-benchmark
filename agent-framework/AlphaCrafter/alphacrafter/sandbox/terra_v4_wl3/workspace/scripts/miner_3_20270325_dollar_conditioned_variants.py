import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).sort_values('date').set_index('date').close
prices=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in assets}).sort_index(); prices=prices[prices.index<=cut]
dr=dxy.pct_change(3).abs(); med=dr.rolling(60,min_periods=20).median().shift(1); shock=(dr/med).clip(.5,3).reindex(prices.index).ffill()
base=-prices.pct_change(3)
for alpha in [.5,1,1.5,2,3,4]:
 fac=base*(1+alpha*(shock-1)); vals=[]; ds=[]; ns=[]
 fwd=prices.pct_change().shift(-1)
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print(alpha,len(s),round(np.mean(ns),2),f'{s.mean():.6f}',f'{s.mean()/s.std(ddof=1):.6f}',(s>0).mean())

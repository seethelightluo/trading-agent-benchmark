import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-01-27'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]; C={};R={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=cut].set_index('date');C[a]=d.close;R[a]=d.close.pct_change()
r=pd.DataFrame(R);c=pd.DataFrame(C); vol=r.rolling(20,min_periods=15).std(); fac=-vol.sub(vol.median(axis=1),axis=0);fac.to_csv('scripts/miner_2_20270128_lowvol_signal.csv')
for h in [1,5,10]:
 f=c.pct_change(h).shift(-h);v=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.nunique().min()>1:v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(v,index=ds);print('H',h,'dates',len(s),'avgN',np.mean(ns),'IC',s.mean(),'ICIR',s.mean()/s.std(),'hit',(s>0).mean())
print('coverage',fac.notna().sum(axis=1).mean()/15,'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())

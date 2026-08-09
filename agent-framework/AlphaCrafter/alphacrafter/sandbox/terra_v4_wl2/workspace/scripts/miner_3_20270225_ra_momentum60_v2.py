import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 x=pd.read_csv(f);x.date=pd.to_datetime(x.date);D[os.path.basename(f)[:-4]]=x.sort_values('date').set_index('date').close
p=pd.DataFrame(D).sort_index(); cutoff=pd.Timestamp('2027-02-25'); p=p.loc[:cutoff]; r=p.pct_change()
# medium horizon risk adjusted momentum: 60d return, penalize recent 10d reversal, scaled by 30d vol
sig=(r.rolling(60).sum()-0.5*r.rolling(10).sum())/(r.rolling(30).std()+1e-9)
for h in [1,5,10]:
 fr=p.shift(-h)/p-1;ics=[];ns=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(ics);print(h,len(a),np.mean(ns),np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12),np.mean(a>0))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15'),('2026-07-16','2027-02-25')]:
 fr=p.shift(-1)/p-1;aa=[]
 for dt in p.loc[lo:hi].index:
  z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(lo,hi,len(aa),np.mean(aa) if aa else None)
sig.to_csv('../persistent/factor_signals_miner_3_20270225_ra_momentum60_v2.csv',index_label='date')

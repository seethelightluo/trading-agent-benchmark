import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2030-02-06'); a={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.split('/')[-1][:-4];d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index();a[s]=d.loc[:cut,'close']
p=pd.concat(a,axis=1).sort_index();r=p.pct_change()
# Reversal of recent 5d return, normalized by downside semideviation (20d), with mild 60d drawdown penalty.
down=r.where(r<0).rolling(20,min_periods=10).std(); dd=p/p.rolling(60,min_periods=30).max()-1
sig=(-(r.rolling(5,min_periods=5).sum())/(down*np.sqrt(20))).mul((1+dd.abs().clip(0,0.5)),axis=0).shift(1)
def run(y):
 z=[];ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 z=np.array(z);return z,np.mean(ns)
for h in [1,5,10]:
 x,n=run(r.shift(-h).rolling(h).sum() if h>1 else r.shift(-1));print('h%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f'%(h,len(x),n,np.mean(x),np.mean(x)/(np.std(x,ddof=1)+1e-12)*np.sqrt(len(x)),np.mean(x>0)))
x,n=run(r.shift(-1));print('recent250 IC=%.6f ICIR=%.6f'%(np.mean(x[-250:]),np.mean(x[-250:])/(np.std(x[-250:],ddof=1)+1e-12)*np.sqrt(250)))

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2026-12-03'); D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'];D[s]=d[d.index<=cut]
P=pd.DataFrame(D).sort_index();R=P.pct_change();v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill(); vz=(v-v.rolling(60,min_periods=40).mean())/v.rolling(60,min_periods=40).std()
for name,f in [('vix_cond_rev2',-R.rolling(2).sum()),('vix_cond_rev5',-R.rolling(5).sum())]:
 # crisis: reversal, calm: mild continuation, based on lagged VIX z
 f=f.mul(np.where(vz.values[:,None]>0,1,-.25)); rows=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],R.shift(-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:rows.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.array(rows);print(name,len(a),a.mean(),a.mean()/a.std(),(a>0).mean(),'turn',f.rank(pct=True,axis=1).diff().abs().mean(1).mean())

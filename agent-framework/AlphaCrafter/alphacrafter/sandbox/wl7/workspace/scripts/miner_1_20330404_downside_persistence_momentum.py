import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index()
r=P.pct_change()
# Candidate: medium-term momentum scaled by downside risk, with a persistence filter.
# All rolling values are known at decision date; forward returns are strictly shifted.
down=r.where(r<0).rolling(40,min_periods=20).std()
ret20=P.pct_change(20)
persist=(r.rolling(20,min_periods=15).mean()>0).astype(float)
f=ret20/(down+1e-8)*persist
f=f.replace([np.inf,-np.inf],np.nan)
f=f.sub(f.mean(axis=1),axis=0)
cut=pd.Timestamp('2033-03-31'); f=f.loc[:cut]
fw=P.shift(-10)/P-1
print('cutoff',cut.date(),'dates',len(f),'N',len(U),'coverage',f.notna().sum().sum()/(len(f)*len(U)))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals)
 print('H%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.3f'%(h,len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)))
for i,g in enumerate(np.array_split(f.index,3),1):
 a=[]
 for dt in g:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('third',i,'n',len(a),'H10IC',np.mean(a) if a else np.nan)
f.stack().rename('signal').reset_index().to_csv('scripts/miner_1_20330404_downside_persistence_momentum_signal.csv',index=False)

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=P.pct_change(); vol=r.rolling(30,min_periods=15).std(); mom=P.pct_change(10)
# volatility-adjusted momentum with a market breadth penalty: rank momentum remains interpretable
breadth=(r.rolling(20,min_periods=15).sum()>0).mean(axis=1)
f=mom/vol.replace(0,np.nan)
f=f.sub(f.mean(axis=1),axis=0)
cut=pd.Timestamp('2032-04-29'); f=f.loc[:cut]
res=[]
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; vals=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(vals);res.append((h,len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)))
print('cutoff',cut.date(),'dates',len(f),'N',len(U),'coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for x in res:print('H%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.3f'%x)
fw=P.shift(-10)/P-1
for i,g in enumerate(np.array_split(f.index,3),1):
 a=[]
 for dt in g:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('third',i,'n',len(a),'H10IC',np.mean(a) if a else np.nan)
f.stack().rename('signal').reset_index().to_csv('scripts/miner_1_20320503_vol_momentum_signal.csv',index=False)

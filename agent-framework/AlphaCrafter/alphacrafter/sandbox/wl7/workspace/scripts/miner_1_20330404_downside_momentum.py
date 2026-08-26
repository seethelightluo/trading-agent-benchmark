import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=P.pct_change()
down=r.where(r<0).rolling(40,min_periods=20).std(); f=P.pct_change(20)/(down+1e-8)
f=f.replace([np.inf,-np.inf],np.nan); f=f.sub(f.mean(axis=1),axis=0); cut=pd.Timestamp('2033-03-31'); f=f.loc[:cut]
def calc(h):
 fw=P.shift(-h)/P-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.array(a); return len(a),np.mean(ns),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0),a
print('cutoff',cut.date(),'dates',len(f),'N',len(U),'coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [5,10,20]:
 x=calc(h); print('H%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.3f'% (h,*x[:5]))
x=calc(10)[-1]
for i,a in enumerate(np.array_split(x,3),1): print('IC third',i,len(a),np.mean(a))
f.stack().rename('signal').reset_index().to_csv('scripts/miner_1_20330404_downside_momentum_signal.csv',index=False)

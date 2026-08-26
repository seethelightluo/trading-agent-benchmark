import numpy as np,pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-08-13'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=P.pct_change(); raw=-(P.shift(1)/P.shift(21)-1)/R.shift(1).rolling(20,min_periods=15).std()
disp=R.shift(1).std(axis=1).where(R.shift(1).notna().sum(axis=1)>=8)
q55=disp.rolling(252,min_periods=126).quantile(.55).shift(1); q70=disp.rolling(252,min_periods=126).quantile(.70).shift(1)
gate=(disp>q55)&(disp<=q70)
sig=raw.mul(gate.astype(float),axis=0).replace([np.inf,-np.inf],np.nan)
print('end',end.date(),'rows',len(P),'active',round(gate.mean(),4))
for h in [5,10,20,40]:
 f=P.shift(-h)/P-1; out=[]; ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1: out.append(spearmanr(z.x,z.y).statistic); ns.append(len(z))
 q=pd.Series(out).dropna(); print(h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
sig.to_csv('scripts/miner_3_20290813_disp55_70_reversal20_signal.csv')
# regime halves
for label,mask in [('early',P.index<='2026-12-31'),('late',P.index>='2027-01-01')]:
 f=P.shift(-20)/P-1; out=[]
 for dt in P.index[mask]:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1: out.append(spearmanr(z.x,z.y).statistic)
 q=pd.Series(out).dropna(); print(label,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))

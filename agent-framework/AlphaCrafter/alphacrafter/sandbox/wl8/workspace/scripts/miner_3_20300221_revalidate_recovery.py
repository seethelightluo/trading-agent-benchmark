import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2030-02-20'
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:cut]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std().shift(1); base=-(P.shift(1)/P.rolling(120,min_periods=80).max().shift(1)-1)/vol; tr=P.shift(1)/P.shift(61)-1; f=base*(1+10*tr.clip(lower=0)); fw=P.shift(-10)/P-1
out=[]
for d in P.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:
  c=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(c):out.append((d,c,len(z)))
R=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); x=R.ic
print('dates',len(x),'start',R.index.min(),'end',R.index.max(),'avg_n',round(R.n.mean(),2),'coverage',round(R.n.mean()/15,4));print('IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4));print('turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
for label,sl in [('2026',('2026-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-12-31')),('2029',('2029-01-01','2029-12-31')),('r360',('2029-02-20','2030-02-20')),('r180',('2029-08-20','2030-02-20'))]:
 q=R.loc[sl[0]:sl[1],'ic'].dropna();print(label,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
for h in [5,20]:
 q=[];fw2=P.shift(-h)/P-1
 for d in P.index:
  z=pd.concat([f.loc[d],fw2.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q).dropna();print('decay',h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20300221_recovery_boost_signal.csv',index=False);R.reset_index().to_csv('scripts/miner_3_20300221_recovery_boost_ic.csv',index=False)

import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2030-03-20'
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:CUT]
r=P.pct_change(); vol=r.rolling(20,min_periods=15).std().shift(1)
raw=-(P.shift(1)/P.rolling(120,min_periods=80).max().shift(1)-1)/vol
trend=P.shift(1)/P.shift(61)-1; f=raw*(1+4*trend.clip(lower=0)); fw=P.shift(-10)/P-1
rows=[]
for d in P.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(ic): rows.append((d,ic,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic
R.to_csv('scripts/miner_3_20300321_recovery_boost4_reval_ic.csv'); f.to_csv('scripts/miner_3_20300321_recovery_boost4_reval_signal.csv')
print('endpoint',P.index.max().date(),'dates',len(R),'start',R.index.min().date(),'end',R.index.max().date(),'avg_n',R.n.mean(),'coverage',R.n.mean()/15)
print('full',x.mean(),x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for label,sub in [('2026',R.loc['2026']),('2027-28',R.loc['2027':'2028']),('2029',R.loc['2029']),('2030',R.loc['2030']),('recent360',R.tail(360)),('recent180',R.tail(180))]: print(label,len(sub),sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1) if len(sub)>1 else np.nan)
for h in [5,10,20]:
 fw2=P.shift(-h)/P-1; q=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw2.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q).dropna(); print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))

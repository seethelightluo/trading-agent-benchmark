import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2030-05-15'
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:CUT]
r20=P.shift(1)/P.shift(21)-1; r120=P.shift(1)/P.shift(121)-1
vol=P.pct_change().rolling(40,min_periods=20).std().shift(1)
# Relative, volatility-normalized medium momentum with a long-trend agreement boost.
rel=r20-r20.median(axis=1).to_numpy()[:,None]
agree=np.where(np.sign(r20)==np.sign(r120),1.25,0.75)
f=(rel/vol*agree).clip(-8,8)
fw=P.shift(-10)/P-1; rows=[]
for d in P.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): rows.append((d,q,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic
f.to_csv('scripts/miner_1_20300516_regime_relative_momentum_signal.csv'); R.to_csv('scripts/miner_1_20300516_regime_relative_momentum_ic.csv')
print('dates',len(R),'start',R.index.min().date(),'end',R.index.max().date(),'avg_n',round(R.n.mean(),3),'coverage',round(R.n.mean()/15,4))
print('full',round(x.mean(),6),'icir',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4),'turn',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
for lab,s in [('2026',R.loc['2026']),('2027-28',R.loc['2027':'2028']),('2029',R.loc['2029']),('2030',R.loc['2030']),('r360',R.tail(360)),('r180',R.tail(180))]: print(lab,len(s),round(s.ic.mean(),6),round(s.ic.mean()/s.ic.std(ddof=1),6))
for h in [5,20]:
 fw2=P.shift(-h)/P-1; q=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw2.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q).dropna(); print('decay',h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))

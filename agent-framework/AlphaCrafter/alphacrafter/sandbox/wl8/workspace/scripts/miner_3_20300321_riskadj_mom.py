import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT='2030-03-20'
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:CUT]
r=P.pct_change(); v=r.rolling(20,min_periods=15).std().shift(1)
# lagged medium-term trend persistence, volatility adjusted and bounded to reduce crypto domination
mom=P.shift(1)/P.shift(61)-1
f=(mom/v).clip(-8,8)
rows=[]; fw=P.shift(-10)/P-1
for d in P.index:
 z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): rows.append((d,q,len(z)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=R.ic
R.to_csv('scripts/miner_3_20300321_riskadj_mom_ic.csv'); f.to_csv('scripts/miner_3_20300321_riskadj_mom_signal.csv')
print('dates',len(R),'start',R.index.min().date(),'end',R.index.max().date(),'avg_n',R.n.mean(),'coverage',R.n.mean()/15)
print('full',x.mean(),x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for lab,s in [('2026',R.loc['2026']),('2027-28',R.loc['2027':'2028']),('2029',R.loc['2029']),('2030',R.loc['2030']),('r360',R.tail(360)),('r180',R.tail(180))]: print(lab,len(s),s.ic.mean(),s.ic.mean()/s.ic.std(ddof=1))
for h in [5,20]:
 fw2=P.shift(-h)/P-1;q=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw2.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q).dropna();print('decay',h,len(q),q.mean(),q.mean()/q.std(ddof=1))

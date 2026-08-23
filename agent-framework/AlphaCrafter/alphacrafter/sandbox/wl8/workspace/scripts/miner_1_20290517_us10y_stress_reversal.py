import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2029-05-16'
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'] for a in U},axis=1).sort_index().loc[:END]
y=pd.read_csv('../persistent/stock_data/US10Y.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].reindex(P.index).ffill()
# Yield-shock-conditioned cross-sectional reversal. The multiplier is formed only from t-1 data.
r=P.pct_change(); shock=y.pct_change(5).rolling(60,min_periods=40).mean() # baseline unused, retained for alignment
z=(y-y.rolling(60,min_periods=40).mean())/y.rolling(60,min_periods=40).std()
# elevated yield z-score amplifies reversal, low yield stress suppresses it
f=(-r.rolling(10,min_periods=10).sum().shift(1)).mul((1+0.6*z.shift(1)).clip(.35,2.2),axis=0)
rows=[]
for dt in P.index:
 q=pd.concat([f.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
 if len(q)>=8: rows.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,len(q)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def st(x): return (len(x),round(x.n.mean(),2),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(ddof=1),6),round((x.ic>0).mean(),4))
print('period',R.index.min(),R.index.max(),'dates',len(R),'avg_n',R.n.mean(),'coverage',R.n.sum()/(len(R)*15))
for name,sl in [('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent180',slice('2028-10-15',END)),('full',slice(None))]: print(name,st(R.loc[sl]))
for h in [1,5,10,20]:
 a=[]
 for dt in P.index:
  q=pd.concat([f.loc[dt],(P.shift(-h)/P-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 x=pd.Series(a);print('decay',h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20290517_us10y_stress_reversal_signal.csv',index=False)

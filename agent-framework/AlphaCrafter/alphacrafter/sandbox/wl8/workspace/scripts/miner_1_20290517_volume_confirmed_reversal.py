import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2029-05-16'
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for a in U}
P=pd.concat({a:D[a].close for a in U},axis=1).sort_index().loc[:END]; V=pd.concat({a:D[a].volume for a in U},axis=1).reindex(P.index)
r=P.pct_change(); ret=(-r.rolling(10,min_periods=10).sum().shift(1));
# Volume-confirmed reversal: emphasize reversals occurring with unusually high lagged turnover.
vr=V.shift(1)/V.shift(1).rolling(60,min_periods=40).median(); f=ret*vr.clip(.5,3)
rows=[]
for dt in P.index:
 q=pd.concat([f.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
 if len(q)>=8: rows.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,len(q)))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def st(x):return len(x),round(x.n.mean(),2),round(x.ic.mean(),6),round(x.ic.mean()/x.ic.std(ddof=1),6),round((x.ic>0).mean(),4)
print('period',R.index.min(),R.index.max(),'dates',len(R),'avg_n',R.n.mean(),'coverage',R.n.sum()/(len(R)*15))
for k,s in [('2026',slice('2026','2026')),('2027-28',slice('2027','2028')),('recent180',slice('2028-10-15',END)),('full',slice(None))]:print(k,st(R.loc[s]))
for h in [1,5,10,20]:
 x=[]
 for dt in P.index:
  q=pd.concat([f.loc[dt],(P.shift(-h)/P-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:x.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 x=pd.Series(x);print('decay',h,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20290517_volume_confirmed_reversal_signal.csv',index=False)

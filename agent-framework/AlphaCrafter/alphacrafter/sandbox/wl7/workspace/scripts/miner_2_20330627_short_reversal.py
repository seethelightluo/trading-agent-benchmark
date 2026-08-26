import pandas as pd,numpy as np
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-06-26'); P={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut]; P[a]=d.close
P=pd.DataFrame(P); r=P.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Contrarian short-horizon move, risk normalized, with a 1-session information lag.
fac=(-P.pct_change(5)/(vol*np.sqrt(20))).shift(1)
fw={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
print('dates',len(fac),'assets',len(assets),'coverage',fac.notna().mean().mean())
for h in fw:
 vals=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(len(z))
 x=np.array(vals); print('H',h,'n',len(x),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f thirds'% (x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)),[round(q.mean(),6) for q in np.array_split(x,3)])
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
out=[{'date':dt.date(),'asset':a,'signal':fac.loc[dt,a]} for dt in fac.index for a in assets]
pd.DataFrame(out).to_csv('scripts/miner_2_20330627_short_reversal_signal.csv',index=False)

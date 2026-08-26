import pandas as pd,numpy as np
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2033-06-12')
P={}; V={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').loc[:cut]
 r=d.close.pct_change(); P[a]=d.close; V[a]=r.rolling(20,min_periods=15).std().shift(1)
for h in [1,5,10,20]:
 arr=[]
 fw=pd.DataFrame({a:P[a].shift(-h)/P[a]-1 for a in assets}); fac=pd.DataFrame({a:-V[a] for a in assets})
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):arr.append(q)
 x=np.array(arr); print('H',h,'n',len(x),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)))
print('coverage',fac.notna().mean().mean())
out=[]
for dt in fac.index:
 for a in assets:out.append({'date':dt.date(),'asset':a,'signal':fac.loc[dt,a]})
pd.DataFrame(out).to_csv('scripts/miner_2_20330613_lowvol_signal.csv',index=False)

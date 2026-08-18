import os, numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2035-11-22'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change()
# Lagged 20-session return divided by preceding 40-session cumulative downside path.
down_path=(-R.where(R<0,0)).rolling(40,min_periods=20).sum()
F=P.pct_change(20).shift(1).div(down_path.shift(1)+1e-12)
ics=[]; dates=[]; cov=[]; dec={1:[],5:[],10:[],20:[]}
for i in range(65,len(P)-20):
 for h in dec:
  z=pd.concat([F.iloc[i],P.shift(-h).iloc[i]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): dec[h].append(q)
 z=pd.concat([F.iloc[i],P.shift(-10).iloc[i]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(q): ics.append(q); dates.append(P.index[i]); cov.append(len(z)/len(U))
a=pd.Series(ics,index=dates)
print('cutoff',cutoff.date(),'dates',len(a),'instruments',len(D),'avg_inst',round(np.mean(cov)*len(U),2),'coverage',round(np.mean(cov),4))
for h,v in dec.items():
 x=pd.Series(v); print('h',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(),6),'hit',round((x>0).mean(),4),'n',len(x))
print('recent252 IC/ICIR',round(a.iloc[-252:].mean(),6),round(a.iloc[-252:].mean()/a.iloc[-252:].std(),6))
print('turnover_proxy',round(F.rank(axis=1).diff().abs().mean().mean()/len(U),6))
F.to_csv('scripts/miner_1_20351123_downside_efficiency_signal.csv')

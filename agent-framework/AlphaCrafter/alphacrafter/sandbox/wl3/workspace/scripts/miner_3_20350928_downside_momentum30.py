import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2035-09-27'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change()
# shorter downside risk estimate improves responsiveness and coverage while retaining 20d momentum
for w in [30]:
 down=R.where(R<0).rolling(w,min_periods=15).std()
 F=P.pct_change(20).shift(1).div(down.shift(1)*np.sqrt(20)+1e-12)
 print('dates',len(P),'instruments',len(D),'cutoff',P.index.max(),'window',w)
 for h in [1,3,5,10,20]:
  vals=[]; cov=[]; dates=[]
  for i in range(65,len(P)-h):
   z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
   if len(z)>=8:
    q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(q): vals.append(q); cov.append(len(z)/len(U)); dates.append(P.index[i])
  a=pd.Series(vals); recent=a.iloc[-252:] if len(a)>252 else a
  print('h',h,'n',len(a),'all_ic',round(a.mean(),6),'all_icir',round(a.mean()/a.std(),4),'recent_n',len(recent),'recent_ic',round(recent.mean(),6),'recent_icir',round(recent.mean()/recent.std(),4),'hit',round((recent>0).mean(),4),'cov',round(np.mean(cov),4))
 F.index.name='date';F.to_csv('scripts/miner_3_20350928_downside_momentum30_signal.csv')

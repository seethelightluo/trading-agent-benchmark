import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-09-13');D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p);d.date=pd.to_datetime(d.date);D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill();R=P.pct_change();down=R.where(R<0).rolling(60,min_periods=30).std();F=P.pct_change(20).shift(1).div(down.shift(1)*np.sqrt(20)+1e-12)
print('dates',len(P),'instruments',len(D),'cutoff',P.index.max())
for h in [1,3,5,10,20]:
 a=[];cov=[];turn=[]
 for i in range(65,len(P)-h):
  x=F.iloc[i];z=pd.concat([x,P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):a.append(q);cov.append(len(z)/len(U));
   if i>65:turn.append((x.reindex(z.index).rank()!=F.iloc[i-1].reindex(z.index).rank()).mean())
 a=pd.Series(a);print(h,len(a),round(a.mean(),8),round(a.mean()/a.std(),5),round((a>0).mean(),4),round(np.mean(cov),4),round(np.mean(turn),4))
F.index.name='date';F.to_csv('scripts/miner_3_20350914_downside_momentum_signal.csv')

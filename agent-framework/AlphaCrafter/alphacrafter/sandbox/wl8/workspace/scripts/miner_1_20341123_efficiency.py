import pandas as pd,numpy as np,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):px[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close']
P=pd.DataFrame(px).sort_index();r=P.pct_change(); lag=r.shift(1)
# directional efficiency: net 20d movement relative to path length, lagged
sig=(P.shift(1).pct_change(20)/(lag.abs().rolling(20,min_periods=15).sum()+1e-10)).clip(-1,1)
fwd=P.shift(-10)/P-1; rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8:rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');a=q.ic.values
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(q)*15));print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turn',sig.rank(pct=True).diff().abs().mean(axis=1).mean())
for dr in [1,-1]:print('direction',dr,'IC',np.mean(a*dr),'ICIR',np.mean(a*dr)/np.std(a*dr,ddof=1))
for n in [365,750,1260]:
 x=q.tail(n).ic.values;print('recent',n,x.mean(),x.mean()/x.std(ddof=1))
for h in [1,5,20]:
 y=P.shift(-h)/P-1;v=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(v))
q.to_csv('scripts/miner_1_20341123_efficiency_ic.csv');sig.to_csv('scripts/miner_1_20341123_efficiency_signal.csv')

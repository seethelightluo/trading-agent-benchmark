import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={};V={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f); x.date=pd.to_datetime(x.date); x=x.set_index('date').sort_index(); P[s]=x.close.astype(float); V[s]=x.volume.astype(float)
p=pd.DataFrame(P).sort_index(); v=pd.DataFrame(V).reindex(p.index); r=p.pct_change()
# volume-confirmed reversal: negative lagged 3d return, stronger after abnormal lagged volume, volatility scaled
ret=r.rolling(3).sum().shift(1); vol=r.shift(1).rolling(20).std(); vr=(v.shift(1)/v.shift(1).rolling(20).median()-1).clip(-.8,3)
sig=(-ret/(vol*np.sqrt(3)))*(1+0.35*vr.clip(lower=0))
forward=p.shift(-1)/p-1
out=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],forward.loc[dt]],axis=1).dropna()
 if len(z)>=8: out.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(out,columns=['date','n','ic']).set_index('date')
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/(len(q)*15)); print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean()); print('turnover',sig.rank(axis=1).diff().abs().mean(axis=1).mean()/15)
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 x=q.loc[a:b]; print(a,len(x),x.ic.mean(),x.ic.mean()/x.ic.std() if len(x)>1 else np.nan)
for h in [5,10,20]:
 f=p.shift(-h)/p-1; zlist=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:zlist.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(zlist),len(zlist))
sig.to_csv('scripts/miner_2_20270507_volume_confirmed_reversal_signal.csv')

import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close
P=pd.concat(D,axis=1).sort_index(); r=P.pct_change(); vr=r.rolling(5).std()/r.rolling(60).std()
sig=(P/P.shift(10)-1).where(vr>=1,(P/P.shift(10)-1)*vr).replace([np.inf,-np.inf],np.nan)
ics=[]; rows=[]; turnovers=[]
for i in range(60,len(P)-10):
 a=sig.iloc[i]; f=P.iloc[i+10]/P.iloc[i]-1; z=pd.concat([a,f],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); rows.append(len(z))
  if i>60:
   q=pd.concat([sig.iloc[i],sig.iloc[i-1]],axis=1).dropna(); turnovers.append(np.mean(np.sign(q.iloc[:,0])!=np.sign(q.iloc[:,1])))
a=np.array(ics); print('dates',len(a),'avg_inst',np.mean(rows),'coverage',np.mean(rows)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',np.mean(a>0),'turnover',np.nanmean(turnovers))
for n in [180,360]:
 b=a[-n:]; print('recent',n,b.mean(),b.mean()/b.std())
for h in [1,5,10]:
 q=[]
 for i in range(60,len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,len(q),np.nanmean(q),np.nanmean(q)/np.nanstd(q))

import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cap=pd.Timestamp('2031-12-10'); px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close.replace(0,np.nan)
P=pd.DataFrame(px).loc[:cap]; r=np.log(P).diff(); m60=np.log(P/P.shift(60)); m10=np.log(P/P.shift(10)); dn=r.where(r<0).rolling(40,min_periods=20).std()*np.sqrt(40); vv=r.rolling(20,min_periods=15).std()*np.sqrt(20)
F=(-(m60-.70*m10)/(dn+.5*vv+1e-8)).shift(1); rows=[]; sig=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],np.log(P.shift(-20)/P).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z))); sig += [(dt,a,float(v)) for a,v in F.loc[dt].dropna().items()]
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=res.ic
print('candidate exhaustion_contrarian_updated'); print('obs',len(q),'avgN',round(res.n.mean(),2),'coverage',round(res.n.mean()/15,4),'IC20',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [1,5,10,20]:
 yy=np.log(P.shift(-h)/P); zc=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: zc.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,round(np.nanmean(zc),6),len(zc))
for a,b in [('2026','2028'),('2029','2030'),('2031','2031')]:
 x=q[(q.index>=a)&(q.index<=b)]; print('regime',a,b,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else np.nan)
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20311211_exhaustion_contrarian_signal.csv',index=False); print('artifact',len(sig))

import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cap=pd.Timestamp('2031-11-12'); px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); px[s]=d.close.replace(0,np.nan)
P=pd.DataFrame(px).loc[:cap]; r=np.log(P).diff(); vol=r.rolling(20,min_periods=15).std(); mom60=np.log(P/P.shift(60));
# contrarian short shock, gated to assets with positive medium trend; lagged
F=((-r.rolling(5).sum()/vol)*((mom60>0).astype(float))).shift(1); Y=np.log(P.shift(-10)/P); rows=[]; sig=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  sig += [(dt,a,float(v)) for a,v in F.loc[dt].dropna().items()]
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for name,mask in [('all',res.index>=res.index.min()),('2026-28',(res.index>='2026-01-01')&(res.index<'2029-01-01')),('2029-30',(res.index>='2029-01-01')&(res.index<'2031-01-01')),('2031YTD',res.index>='2031-01-01')]:
 q=res.loc[mask].ic; print(name,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6),round((q>0).mean(),4))
print('obs',len(res),'avgN',round(res.n.mean(),2),'coverage',round(res.n.sum()/(len(res)*15),4))
for h in [1,5,10,20]:
 yy=np.log(P.shift(-h)/P); q=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,round(np.nanmean(q),6),len(q))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20311113_shock_reversal_signal.csv',index=False)

import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); px[s]=d['close'].replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); r=np.log(P).diff(); mom=np.log(P/P.shift(20))
bread=((r>0).rolling(40,min_periods=30).mean()-0.5)*2
F=(mom*bread).shift(1); rows=[]; sig=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],np.log(P.shift(-10)/P).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  for a,v in F.loc[dt].dropna().items(): sig.append((dt,a,float(v)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for name,mask in [('all',res.index>=res.index.min()),('2026-28',(res.index>='2026-01-01')&(res.index<'2029-01-01')),('2029-30',(res.index>='2029-01-01')&(res.index<'2031-01-01')),('2031YTD',res.index>='2031-01-01')]:
 q=res.loc[mask,'ic'].dropna(); print(name,'dates',len(q),'meanIC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('obs',len(res),'avgN',round(res.n.mean(),2),'coverage',round(res.n.sum()/(len(res)*15),4))
for h in [1,5,10,20]:
 yy=np.log(P.shift(-h)/P); rr=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,round(np.nanmean(rr),6),len(rr))
out=pd.DataFrame(sig,columns=['date','symbol','signal']); out.to_csv('scripts/miner_3_20311113_breadth_confirmed_momentum_signal.csv',index=False); print('signal rows',len(out)); print(res.tail(3).to_string())

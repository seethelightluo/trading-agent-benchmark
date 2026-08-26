import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).set_index('date'); px[s]=d['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
up=r.clip(lower=0).rolling(60,min_periods=40).std(); dn=(-r.clip(upper=0)).rolling(60,min_periods=40).std()
f=(up/(dn+1e-8)).shift(1).replace([np.inf,-np.inf],np.nan).clip(0.2,5)
fr=p.shift(-10)/p-1; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=q.ic.values
print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.sum()/(len(q)*15))
print('IC10',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(rr),len(rr))
for n in [365,750,1260]:
 aa=q.tail(n).ic.values;print('recent',n,len(aa),aa.mean(),aa.mean()/aa.std(ddof=1))
q.to_csv('scripts/miner_1_20341123_downside_asymmetry_ic.csv'); f.to_csv('scripts/miner_1_20341123_downside_asymmetry_signal.csv')

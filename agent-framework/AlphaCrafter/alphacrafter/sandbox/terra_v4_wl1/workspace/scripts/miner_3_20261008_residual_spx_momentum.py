import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-10-08')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index()['close'].astype(float); px[s]=d[d.index<=cut]
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); bench=R['SPX']
beta=R.rolling(60,min_periods=40).cov(bench).div(bench.rolling(60,min_periods=40).var(),axis=0)
res=R-beta.mul(bench,axis=0); f=res.rolling(40,min_periods=30).sum()
for h in [1,3,5,10]:
 z=[]
 for dt in R.index:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1
  q=pd.DataFrame({'f':f.loc[dt],'y':y}).dropna()
  if len(q)>=8: z.append((dt,q.f.corr(q.y),len(q)))
 a=pd.DataFrame(z,columns=['date','ic','n']).set_index('date').ic.dropna(); ns=pd.DataFrame(z,columns=['date','ic','n']).set_index('date').n
 print('H',h,'dates',len(a),'avgN',ns.mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
 if h==1:
  for yr,g in a.groupby(a.index.year): print('year',yr,'IC',g.mean(),'n',len(g))
p=f.rank(axis=1,pct=True); print('coverage',f.notna().sum().sum()/(f.shape[0]*f.shape[1]),'turnover',p.diff().abs().mean().mean())
print('corr with raw 20d momentum',f.stack().corr(R.rolling(20).sum().stack()))

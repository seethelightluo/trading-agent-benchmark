import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in S:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): px[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index().loc[:'2027-10-06']
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); f0=(-r.rolling(5,min_periods=5).sum()/r.rolling(20,min_periods=15).std()).clip(-8,8)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill(); stress=(vix/vix.rolling(60,min_periods=40).median()).clip(.5,2)
fwd=p.shift(-1)/p-1
for name,g in {'plain':1,'stress_amp':stress.pow(.5),'stress_damp':stress.pow(-.5)}.items():
 f=f0.mul(g,axis=0).shift(1).ewm(span=3,min_periods=3,adjust=False).mean(); rows=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=o.ic
 print(name,'dates',len(o),'avgN',o.n.mean(),'coverage',o.n.sum()/(len(o)*15),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
 for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
  x=o.loc[a:b].ic; print(' regime',a,len(x),x.mean(),x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
 for h in [3,5,10]:
  yy=p.shift(-h)/p-1; zics=[]
  for d in f.index:
   z=pd.concat([f.loc[d],yy.loc[d]],axis=1).dropna()
   if len(z)>=8:zics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  x=pd.Series(zics); print(' h',h,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'dates',len(x))

import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUTOFF=pd.Timestamp('2028-12-14')
def load(s):
 for root in ['../persistent/stock_data/','../persistent/index_data/']:
  f=root+s+'.csv'
  if os.path.exists(f): return pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()['close'].astype(float)
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}; P=pd.DataFrame(D).sort_index().loc[:CUTOFF].ffill(); R=np.log(P).diff()
# multi-horizon risk-adjusted trend: short/medium/long returns, penalized by realized vol
v=R.rolling(20,min_periods=15).std()*np.sqrt(252)
F=(0.25*R.rolling(5,min_periods=5).sum()+0.45*R.rolling(20,min_periods=15).sum()+0.30*R.rolling(60,min_periods=40).sum())/v
# cross-sectional rank to limit scale
F=F.rank(axis=1,pct=True)
for h in [1,5,10,20]:
 fw=np.log(P.shift(-h)/P); rows=[]
 for dt in F.index[:-h]:
  z=pd.concat([F.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna(); ic=r.ic.mean(); ir=ic/r.ic.std(ddof=1)
 print('horizon',h,'dates',len(r),'nmean',round(r.n.mean(),2),'coverage',round(F.loc[r.index].notna().mean().mean(),4),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((r.ic>0).mean(),4))
 for nm,q in [('recent252',r.tail(252)),('2027+',r.loc['2027-01-01':]),('2028YTD',r.loc['2028-01-01':])]:
  if len(q)>1: print(nm,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1),6))
print('assets',len(D),'dates',len(P),'cutoff',P.index.max().date())

import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15'); D={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index();D[s]=d[d.index<=end]
C=pd.DataFrame({s:d.close for s,d in D.items()});H=pd.DataFrame({s:d.high for s,d in D.items()});L=pd.DataFrame({s:d.low for s,d in D.items()});O=pd.DataFrame({s:d.open for s,d in D.items()}); F=-(2*(C-L)/(H-L).replace(0,np.nan)-1)
def forward(s,dt,h):
 ix=D[s].index.get_loc(dt);j=ix+h
 return np.nan if j>=len(D[s]) else D[s].iloc[j].close/D[s].iloc[ix].close-1
for h in [1,5,10,20]:
 a=[]; ns=[]
 for dt in F.index:
  x=[];y=[]
  for s in U:
   if pd.notna(F.loc[dt,s]) and dt in D[s].index:
    q=forward(s,dt,h)
    if pd.notna(q):x.append(F.loc[dt,s]);y.append(q)
  if len(x)>=8:a.append(pd.Series(x).corr(pd.Series(y),method='spearman'));ns.append(len(x))
 a=np.array(a);print(h,len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
# correlation against exact current factor expressions, pooled observations (library max)
for nm,X in [('peer',None),('reversal',-(C/C.shift(5)-1)),('ram',(C/C.shift(20)-1)/C.pct_change().rolling(60).std())]:
 if X is None:
  r=C.pct_change(5); X=pd.DataFrame(index=C.index,columns=C.columns)
  for dt,row in r.iterrows():
   for s in U:X.loc[dt,s]=row.drop(s).median()
 z=pd.concat([F.stack(),X.stack()],axis=1).dropna();print('CORR',nm,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z))
print('coverage', (F.notna().sum(axis=1)>=8).mean(),'valid dates',sum(F.notna().sum(axis=1)>=8),'meanN',F.notna().sum(axis=1)[F.notna().sum(axis=1)>=8].mean())
rank=F.rank(axis=1,pct=True);print('turn',((rank-rank.shift()).abs().mean(axis=1)).mean())

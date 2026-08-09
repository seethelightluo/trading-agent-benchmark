import pandas as pd,numpy as np,glob
D={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 s=p.split('/')[-1][:-4];x=pd.read_csv(p);x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close
px=pd.concat(D,axis=1).sort_index();r=px.pct_change()
# persistence: signed breadth of recent daily returns, robust trend quality
f=r.gt(0).rolling(20,min_periods=15).mean()-r.lt(0).rolling(20,min_periods=15).mean()
for h in [1,5,10]:
 fr=px.shift(-h)/px-1; vals=[];ns=[];ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(vals,index=ds).dropna();print('h',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 if h==1:print('turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),'cov',f.notna().mean(axis=1).mean())
 for lab,ix in [('early',q.index<'2023-01-01'),('late',q.index>='2023-01-01')]:
  z=q[ix];print(lab,len(z),z.mean(),z.mean()/z.std(ddof=1))
print('corr20mom',f.stack().corr(r.rolling(20).sum().stack()))

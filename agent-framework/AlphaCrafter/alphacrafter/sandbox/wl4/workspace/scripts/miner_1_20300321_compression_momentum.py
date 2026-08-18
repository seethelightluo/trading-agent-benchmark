import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:x=get_stock_daily_data(s,days=2600)
 except:continue
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close.astype(float)
p=pd.concat(D,axis=1).sort_index().ffill();r=np.log(p).diff()
# favors compression plus positive medium trend; standardized cross-section via rank product
comp=-(r.rolling(10).std()/r.rolling(40).std()); mom=np.log(p).diff(20)
f=(comp.rank(axis=1,pct=True)*mom.rank(axis=1,pct=True)).shift(1); out=[]
for d in f.index:
 q=pd.concat([f.loc[d],(np.log(p).shift(-10)-np.log(p)).loc[d]],axis=1).dropna()
 if len(q)>=8:out.append((d,len(q),q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
z=pd.DataFrame(out,columns=['d','n','ic']).set_index('d')
for lab,q in [('full',z),('recent250',z.tail(250)),('early',z.iloc[:len(z)//3]),('mid',z.iloc[len(z)//3:2*len(z)//3]),('late',z.iloc[2*len(z)//3:])]:
 print(lab,len(q),round(q.n.mean(),2),q.n.min(),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(ddof=1)*np.sqrt(252),6),round((q.ic>0).mean(),4))
print('coverage',round(f.notna().sum().sum()/(f.shape[0]*f.shape[1]),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))

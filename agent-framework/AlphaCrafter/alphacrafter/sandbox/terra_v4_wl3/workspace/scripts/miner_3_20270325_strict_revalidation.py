import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); C={}
for p in glob.glob('../persistent/stock_data/*.csv'):
 s=os.path.basename(p)[:-4]; d=pd.read_csv(p,parse_dates=['date']).sort_values('date').set_index('date'); C[s]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r3=close.pct_change(3); breadth=(close.pct_change(5)<0).mean(axis=1)
# strict 60-90% stress gate, matching persisted definition
act=((breadth-.60)/.30).clip(0,1)
rel=r3.sub(r3.median(axis=1),axis=0); z=rel.sub(rel.mean(axis=1),axis=0).div(rel.std(axis=1).replace(0,np.nan),axis=0)
fac=(-z).mul(act,axis=0); y=close.pct_change(1).shift(-1)
rows=[]
for dt in fac.index:
 x=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic,len(x)))
df=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('full',len(df),round(df.ic.mean(),7),round(df.ic.mean()/df.ic.std(ddof=1),7),'avgN',round(df.n.mean(),2))
for end in ['2026-09-30','2026-12-31','2027-03-24']:
 q=df.loc[pd.Timestamp(end)-pd.Timedelta(days=183):end]
 print('6m',end,'dates',len(q),'IC %.7f ICIR %.7f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
 q=df[(df.index>=lo)&(df.index<=hi)]; print('regime',lo,len(q),'IC %.7f ICIR %.7f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('coverage',fac.notna().sum(axis=1).mean()/15,'active',(act>0).sum())

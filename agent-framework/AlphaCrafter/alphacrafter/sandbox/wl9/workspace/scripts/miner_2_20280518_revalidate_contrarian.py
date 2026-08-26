import numpy as np,pandas as pd
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-05-17'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().close.astype(float);D[s]=x.loc[:end]
p=pd.DataFrame(D).sort_index();r=p.pct_change(); out=[]
for i in range(21,len(p)-1):
 f=-(p.iloc[i-1].div(p.iloc[i-21])-1)/(r.iloc[i-20:i].std()+.005)
 y=p.iloc[i+1].div(p.iloc[i])-1;q=pd.concat([f,y.rename('y')],axis=1).dropna()
 if len(q)>=8:out.append((r.index[i],len(q),spearmanr(q.iloc[:,0],q.y).statistic))
a=pd.DataFrame(out,columns=['date','n','ic']);print('dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
for nm,cut in [('online','2026-07-16'),('recent252','2027-05-18'),('ytd','2028-01-01')]:
 q=a[a.date>=pd.Timestamp(cut)];print(nm,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))

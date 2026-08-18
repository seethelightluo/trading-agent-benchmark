import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
base='../persistent/stock_data'; syms=sorted(x[:-4] for x in os.listdir(base) if x.endswith('.csv'))
px={}
for s in syms:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); vol5=r.rolling(5,min_periods=5).std(); vol20=r.rolling(20,min_periods=20).std(); vol60=r.rolling(60,min_periods=60).std()
# Relative reversal over 5d, scaled by volatility compression/expansion ratio
f=-(P.pct_change(5))*((vol5/(vol20+1e-12)).clip(0,5))
fr=P.shift(-10)/P.shift(-1)-1; rows=[]
for dt in P.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('universe',len(syms),'dates',len(o),'meanN',o.n.mean(),'coverage',o.n.mean()/len(syms))
print('IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(ddof=1),'hit',(o.ic>0).mean())
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
 q=o.loc[a:b].ic
 if len(q): print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
print('recent',o.tail(60).ic.mean(),o.tail(60).ic.mean()/o.tail(60).ic.std(ddof=1))
rank=f.rank(axis=1,pct=True); rr=[]
for i in range(1,len(rank)):
 z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(z)>=8: rr.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('turnover_proxy',np.nanmean(rr))

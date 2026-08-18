import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2035-11-22'); D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p); d.date=pd.to_datetime(d.date); D[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(D).sort_index().loc[:cutoff].ffill(); R=P.pct_change();
# trend persistence: medium return risk adjusted by fraction of positive daily observations
for w,h in [(15,5),(20,10),(30,10),(40,20)]:
 vol=R.rolling(30,min_periods=20).std(); consistency=(R.gt(0).rolling(w,min_periods=w).mean()-0.5)*2
 F=P.pct_change(w).div(vol+1e-12)*consistency
 vals=[]; cov=[]; dates=[]
 for i in range(50,len(P)-h):
  z=pd.concat([F.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);cov.append(len(z)/15);dates.append(P.index[i])
 a=pd.Series(vals,index=dates); print('w,h',w,h,'dates',len(a),'avgN',round(np.mean(cov)*15,2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'recent252',round(a.tail(252).mean(),6),round(a.tail(252).mean()/a.tail(252).std(ddof=1),6),'turn',round(F.rank(axis=1).diff().abs().mean().mean()/15,6))
 for j,x in enumerate(np.array_split(a,4),1): print(' block',j,round(x.mean(),6),round(x.mean()/x.std(ddof=1),6))

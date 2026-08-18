import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/index_data/'+s+'.csv'
 try: D[s]=pd.read_csv(p)
 except: D[s]=None
P=pd.DataFrame({s:(d.set_index(pd.to_datetime(d.date))['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index(); R=P.pct_change()
mom=R.rolling(10,min_periods=8).sum(); vol=R.rolling(20,min_periods=15).std(); rev=-R.rolling(5,min_periods=4).sum(); F=mom/(vol*np.sqrt(10)+1e-8)+0.25*rev/(vol*np.sqrt(5)+1e-8)
ics=[]; cov=[]; turns=[]
for i in range(len(P)-10):
 z=pd.concat([F.iloc[i],P.pct_change(10).shift(-10).iloc[i]],axis=1).dropna()
 if len(z)>=8: ics.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); cov.append(len(z)/15)
for i in range(1,len(F)):
 q=pd.concat([F.iloc[i-1].rank(pct=True),F.iloc[i].rank(pct=True)],axis=1).dropna()
 if len(q)>=8: turns.append(np.mean(abs(q.iloc[:,1]-q.iloc[:,0])))
a=np.array([x for x in ics if np.isfinite(x)]); ir=float(np.mean(a)/(np.std(a,ddof=1)+1e-12)*np.sqrt(len(a)))
print({'dates':len(a),'average_instruments':round(float(np.mean(np.array(cov)*15)),2),'coverage':round(float(np.mean(cov)),4),'ic':round(float(np.mean(a)),6),'icir':round(ir,4),'hit':round(float(np.mean(a>0)),4),'turnover':round(float(np.mean(turns)),5)})
for n in [1,5,10,20]:
 aa=[]
 for i in range(len(P)-n):
  z=pd.concat([F.iloc[i],P.pct_change(n).shift(-n).iloc[i]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 aa=np.array([x for x in aa if np.isfinite(x)]); di=float(np.mean(aa)/(np.std(aa,ddof=1)+1e-12)*np.sqrt(len(aa)))
 print('decay',n,round(float(np.mean(aa)),6),round(di,4),len(aa))

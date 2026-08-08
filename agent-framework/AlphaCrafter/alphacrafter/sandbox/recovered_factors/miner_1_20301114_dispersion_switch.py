import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in A}
idx=sorted(set.intersection(*[set(x.index) for x in D.values()])); out=[]
for j,t in enumerate(idx):
 if j<70 or j+1>=len(idx):continue
 rs={}; y={}
 for a in A:
  x=D[a]; k=x.index.get_loc(t); c=x.close.values
  if k<21 or k+1>=len(x):continue
  rs[a]=(c[k]/c[k-5]-1,c[k]/c[k-20]-1); y[a]=c[k+1]/c[k]-1
 if len(rs)<8:continue
 disp=np.std([v[1] for v in rs.values()]); hist=[]
 for q in range(k-60,k+1):
  z=[]
  for a in A:
   c=D[a].close.values
   if q>=20:z.append(c[q]/c[q-20]-1)
  if len(z)>=8:hist.append(np.std(z))
 threshold=np.median(hist)
 for name,sign in [('high_reversal',-1),('low_momentum',1)]:
  # high dispersion -> reversal; low dispersion -> momentum
  q=[];f=[]
  for a,(r5,r20) in rs.items():
   s=(-r5 if disp>=threshold else r5) if name=='high_reversal' else (-r5 if disp<threshold else r5)
   q.append(s);f.append(y[a])
  if np.std(q)>0 and np.std(f)>0:out.append((t,name,spearmanr(q,f).statistic,len(q),disp>=threshold))
r=pd.DataFrame(out,columns=['date','mode','ic','n','high']).set_index('date')
print('common dates',len(idx),'assets',len(A))
for m in r['mode'].unique():
 z=r[r['mode']==m].ic;print(m,'dates',len(z),'meanN',r[r['mode']==m].n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean(),'highshare',r[r['mode']==m].high.mean())
 for lab,zz in [('2020-23',z.loc['2020':'2023']),('2024-27',z.loc['2024':'2027']),('2028+',z.loc['2028':]),('latest120',z.tail(120))]:print(lab,len(zz),round(zz.mean(),6),round(zz.mean()/zz.std(ddof=1),6))
 print('coverage cells',r[r['mode']==m].n.sum()/(len(idx)*15))

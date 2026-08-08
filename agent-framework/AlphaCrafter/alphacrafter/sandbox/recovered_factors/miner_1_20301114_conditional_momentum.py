import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date') for a in assets}
idx=sorted(set.intersection(*[set(x.index) for x in D.values()]))
# Conditional cross-asset momentum: 20d return, activated/amplified when cross-sectional dispersion is high.
# All fields use close through t; target is t+1 close-to-close.
rows=[]
for j,t in enumerate(idx):
 if j<25 or j+1>=len(idx): continue
 vals={}; fut={}; rets=[]
 for a in assets:
  x=D[a]; k=x.index.get_loc(t)
  if k<21 or k+1>=len(x): continue
  c=x.close.values
  r20=c[k]/c[k-20]-1
  r5=c[k]/c[k-5]-1
  r1=c[k]/c[k-1]-1
  if np.isfinite(r20) and np.isfinite(r5) and np.isfinite(r1): rets.append(r20)
  vals[a]=(r20,r5,r1)
  fut[a]=c[k+1]/c[k]-1
 if len(rets)<8: continue
 disp=np.std(rets); med=np.median(rets)
 for mode in ['raw','disp_cond','breadth_confirm']:
  q=[]; y=[]
  for a,(r20,r5,r1) in vals.items():
   if a not in fut: continue
   if mode=='raw': s=r20
   elif mode=='disp_cond': s=r20*(1+5*min(disp,0.10))
   else: s=r20*(1 if med*r20>=0 else -0.5)
   q.append(s); y.append(fut[a])
  if len(q)>=8 and np.std(q)>0 and np.std(y)>0:
   rows.append((t,mode,spearmanr(q,y).statistic,len(q),disp))
r=pd.DataFrame(rows,columns=['date','mode','ic','n','disp']).set_index('date')
print('universe',len(assets),'common_dates',len(idx))
for mode in r.mode.unique():
 z=r[r.mode==mode].ic.dropna(); print('\n',mode,'dates',len(z),'meanN',r[r.mode==mode].n.mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 for label,zz in [('2020-23',z.loc['2020':'2023']),('2024-27',z.loc['2024':'2027']),('2028+',z.loc['2028':]),('latest120',z.tail(120))]:
  print(label,len(zz),round(zz.mean(),6),round(zz.mean()/zz.std(ddof=1),6) if len(zz)>1 else np.nan)
 print('coverage',r[r.mode==mode].n.sum()/(len(idx)*15))

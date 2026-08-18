import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s, days=4000)
    except Exception: x=None
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.drop_duplicates('date').set_index('date').sort_index(); D[s]=x['close'].astype(float)
P=pd.DataFrame(D).sort_index(); R=np.log(P/P.shift(1));
for h in [1,5,10,20]:
 out=[]
 for i in range(80,len(P)-h):
  vals={}
  for s in U:
   if s not in P or 'SPX' not in P: continue
   z=pd.concat([R[s].iloc[i-59:i+1],R['SPX'].iloc[i-59:i+1]],axis=1).dropna()
   if len(z)<30: continue
   b=np.cov(z.iloc[:,0],z.iloc[:,1],ddof=1)[0,1]/(np.var(z.iloc[:,1],ddof=1)+1e-12)
   vals[s]=np.log(P[s].iloc[i]/P[s].iloc[i-20])-b*np.log(P['SPX'].iloc[i]/P['SPX'].iloc[i-20])
  f=pd.Series(vals).dropna(); fut=(P.shift(-h).iloc[i]/P.iloc[i]-1).reindex(f.index).dropna()
  if len(f)>=8 and len(fut)>=8: out.append(f.corr(fut,method='spearman'))
 a=np.array(out); print(h,len(a),'IC',round(a.mean(),5),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),5),'hit',round(np.mean(a>0),4),'recent250',round(a[-250:].mean(),5))
print('dates',P.index.min(),P.index.max(),'instruments',len(P.columns),'avg daily coverage',round(P.notna().sum(axis=1).mean(),2))

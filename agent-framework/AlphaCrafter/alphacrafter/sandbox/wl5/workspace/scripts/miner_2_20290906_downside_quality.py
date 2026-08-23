import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    p=Path('../persistent/stock_data')/(s+'.csv')
    if p.exists():
      x=pd.read_csv(p,parse_dates=['date']).sort_values('date'); D[s]=x.set_index('date')['close']
px=pd.DataFrame(D).sort_index().ffill()
r=px.pct_change(); ret20=px.pct_change(20)
down=r.where(r<0,0).rolling(20).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
pos=r.gt(0).rolling(20).mean(); f=ret20/(down*np.sqrt(20)+1e-8)*(0.5+pos)
for h in [5,10,20]:
 rows=[]
 for i in range(25,len(px)-h):
  a=f.iloc[i]; y=px.iloc[i+h]/px.iloc[i]-1; v=pd.concat([a,y],axis=1).dropna()
  if len(v)>=8: rows.append((px.index[i],v.iloc[:,0].corr(v.iloc[:,1]),len(v)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); mean=q.ic.mean(); sd=q.ic.std(ddof=1)
 print(h,'dates',len(q),'meanN',q.n.mean(),'IC',mean,'ICIR',mean/sd*np.sqrt(252),'hit',(q.ic>0).mean(),'coverage',q.n.mean()/15)
 for name,mask in [('2020-24',q.date.dt.year<=2024),('2025-26',q.date.dt.year.isin([2025,2026])),('2027-28',q.date.dt.year.isin([2027,2028])),('2029',q.date.dt.year==2029)]:
  w=q[mask]; print(name,len(w),w.ic.mean(),w.ic.mean()/w.ic.std(ddof=1)*np.sqrt(252) if len(w)>2 else np.nan)
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20290906_downside_quality_signal.csv',index=False); print('artifact',len(out),'dates',out.date.nunique(),'assets',out.symbol.nunique())

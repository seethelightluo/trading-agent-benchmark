import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 c=d.close.astype(float); r=c.pct_change()
 # Directional efficiency: net 30-session move divided by path length, with volatility normalization.
 path=r.abs().rolling(30,min_periods=22).sum(); net=c.pct_change(30)
 vol=r.rolling(30,min_periods=22).std()*np.sqrt(30)
 sig=(net/(path+1e-12) * (1/(vol+1e-12))).shift(1)
 P[s]=pd.DataFrame({'f':sig,'c':c})
rows=[]
for s,x in P.items():
 y=x.c.pct_change(10).shift(-10)
 z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s; rows.append(z.reset_index())
a=pd.concat(rows,ignore_index=True)
out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
  q=g.f.corr(g.y,method='spearman')
  if pd.notna(q): out.append((dt,q,len(g)))
ic=pd.DataFrame(out,columns=['date','ic','n']); q=ic.ic
rank=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True)
print('assets',len(P),'dates',len(ic),'avgN',ic.n.mean(),'coverage',len(a)/(len(set(a.date))*len(U)))
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',rank.diff().abs().mean(axis=1).mean()/2)
for h in [1,5,10,20]:
 vals=[]
 for s,x in P.items():
  y=x.c.pct_change(h).shift(-h); z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s
  for dt,g in z.reset_index().groupby('date'):
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
    c=g.f.corr(g.y,method='spearman')
    if pd.notna(c): vals.append(c)
 print('decay',h,'n',len(vals),'IC',np.mean(vals) if vals else None,'ICIR',np.mean(vals)/np.std(vals,ddof=1) if len(vals)>1 else None)
a.to_csv('scripts/miner_2_20350329_range_efficiency_momentum_panel.csv',index=False)
ic.to_csv('scripts/miner_2_20350329_range_efficiency_momentum_ic.csv',index=False)

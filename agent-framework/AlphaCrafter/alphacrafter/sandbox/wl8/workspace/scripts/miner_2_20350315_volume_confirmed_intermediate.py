import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).set_index('date'); r=d.close.pct_change(); v=d.volume.replace(0,np.nan)
 trend=d.close.pct_change(40); abnormal=(v/v.rolling(60,min_periods=30).median()).clip(.25,4)
 risk=r.rolling(40,min_periods=25).std()*np.sqrt(40)
 sig=(trend*abnormal/risk.replace(0,np.nan)).shift(1)
 P[s]=pd.DataFrame({'f':sig,'c':d.close})
rows=[]
for s,x in P.items():
 y=x.c.pct_change(10).shift(-10); z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s; rows += [z.reset_index()]
a=pd.concat(rows,ignore_index=True)
out=[]
for dt,g in a.groupby('date'):
 if len(g)>=8:
  q=g.f.corr(g.y,method='spearman')
  if pd.notna(q): out.append((dt,q,len(g)))
ic=pd.DataFrame(out,columns=['date','ic','n']); q=ic.ic
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',len(a)/(len(set(a.date))*15))
print('IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
r=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True)
print('turnover',r.diff().abs().mean(axis=1).mean()/2)
for h in [1,5,10,20]:
 vals=[]
 for s,x in P.items():
  y=x.c.pct_change(h).shift(-h); z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s
  for dt,g in z.reset_index().groupby('date'):
   if len(g)>=8:
    c=g.f.corr(g.y,method='spearman')
    if pd.notna(c): vals.append(c)
 print('decay',h,'n',len(vals),'IC',np.mean(vals) if vals else None,'ICIR',np.mean(vals)/np.std(vals,ddof=1) if len(vals)>1 else None)
a.to_csv('scripts/miner_2_20350315_volume_confirmed_intermediate_panel.csv',index=False)
ic.to_csv('scripts/miner_2_20350315_volume_confirmed_intermediate_ic.csv',index=False)

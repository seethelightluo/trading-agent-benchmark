import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date');D[s]=x
P={}
for s,d in D.items():
 rng=(d.high-d.low).replace(0,np.nan); cl=((d.close-d.open)/rng).clip(-1,1); r=d.close.pct_change()
 P[s]=pd.DataFrame({'f':(cl.rolling(10,min_periods=8).mean()/(r.rolling(20,min_periods=15).std()*np.sqrt(10)+1e-12)).shift(1),'c':d.close})
rows=[]
for s,x in P.items():
 y=x.c.pct_change(10).shift(-10);z=pd.concat([x.f,y.rename('y')],axis=1).dropna();z['s']=s;rows.append(z.reset_index())
a=pd.concat(rows);ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8:ics.append((dt,g.f.corr(g.y,method='spearman'),len(g)))
ic=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date').dropna();q=ic.ic
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',len(a)/(len(set(a.date))*15),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
rr=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True);print('turnover',rr.diff().abs().mean(axis=1).mean()/2)
for h in [1,5,10,20]:
 vals=[]
 for s,x in P.items():
  y=x.c.pct_change(h).shift(-h);z=pd.concat([x.f,y.rename('y')],axis=1).dropna();z['s']=s
  for dt,g in z.reset_index().groupby('date'):
   if len(g)>=8: vals.append(g.f.corr(g.y,method='spearman'))
 print('decay',h,np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1))
out=a.pivot(index='date',columns='s',values='f');out.to_csv('scripts/miner_3_20350215_range_pressure_signal.csv');ic.reset_index().to_csv('scripts/miner_3_20350215_range_pressure_ic.csv',index=False)

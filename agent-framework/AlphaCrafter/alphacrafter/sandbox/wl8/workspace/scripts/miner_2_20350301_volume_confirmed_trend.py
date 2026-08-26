import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  D[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')
P={}
for s,d in D.items():
 r=d.close.pct_change(); vol=d.volume.replace(0,np.nan)
 # lagged trend, confirmed by abnormal volume, normalized by recent risk
 trend=d.close.pct_change(20)
 vr=(vol/vol.rolling(40,min_periods=20).median()).clip(.25,4)
 risk=r.rolling(20,min_periods=15).std()*np.sqrt(20)
 f=(trend*vr/risk.replace(0,np.nan)).shift(1)
 P[s]=pd.DataFrame({'f':f,'c':d.close})
rows=[]
for s,x in P.items():
 y=x.c.pct_change(10).shift(-10); z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s; rows.append(z.reset_index())
a=pd.concat(rows,ignore_index=True); rows=[]
for dt,g in a.groupby('date'):
 if len(g)>=8: rows.append((dt,g.f.corr(g.y,method='spearman'),len(g)))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna(); q=ic.ic
print('dates',len(ic),'avgN',ic.n.mean(),'coverage',len(a)/(len(set(a.date))*15),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
r=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True)
print('turnover',r.diff().abs().mean(axis=1).mean()/2)
for h in [1,5,10,20]:
 vals=[]
 for s,x in P.items():
  y=x.c.pct_change(h).shift(-h); z=pd.concat([x.f,y.rename('y')],axis=1).dropna(); z['s']=s
  for dt,g in z.reset_index().groupby('date'):
   if len(g)>=8: vals.append(g.f.corr(g.y,method='spearman'))
 print('decay',h,np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1))
a.to_csv('scripts/miner_2_20350301_volume_confirmed_trend_panel.csv',index=False)
ic.reset_index().to_csv('scripts/miner_2_20350301_volume_confirmed_trend_ic.csv',index=False)

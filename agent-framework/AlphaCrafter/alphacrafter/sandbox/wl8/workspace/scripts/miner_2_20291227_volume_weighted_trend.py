import numpy as np, pandas as pd, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); x=x[x.date<=pd.Timestamp('2029-12-26')].drop_duplicates('date').set_index('date').sort_index(); D[s]=x
px=pd.concat({s:d['close'] for s,d in D.items()},axis=1).sort_index(); vol=pd.concat({s:d['volume'] for s,d in D.items()},axis=1).sort_index(); ret=px.pct_change()
vr=vol.rolling(20,min_periods=10).mean()/vol.rolling(60,min_periods=20).mean(); sig=(ret.rolling(20,min_periods=15).sum()*vr).div(ret.rolling(20,min_periods=15).std()).shift(1)
rows=[]
for dt in sig.index:
 a=sig.loc[dt]; b=(px.shift(-10)/px-1).loc[dt]; z=pd.concat([a,b],axis=1).dropna()
 if len(z)>=8 and dt>=pd.Timestamp('2026-07-16'): rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); rr=sig.rank(axis=1,pct=True); turn=rr.diff().abs().mean(axis=1).mean()
print('dates',len(r),'range',r.index.min(),r.index.max(),'avg_n',r.n.mean(),'coverage',r.n.mean()/15); print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean(),turn))
for name,sub in [('2026',r[r.index.year==2026]),('2027_28',r[(r.index.year>=2027)&(r.index.year<=2028)]),('recent360',r[r.index>=r.index.max()-pd.Timedelta(days=360)]),('recent180',r[r.index>=r.index.max()-pd.Timedelta(days=180)])]: print(name,len(sub),'IC %.6f ICIR %.6f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std()) if len(sub)>2 else 'NA')
for h in [1,5,10,20]:
 ff=px.shift(-h)/px-1; q=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8 and dt>=pd.Timestamp('2026-07-16'):q.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'n',len(q),'IC %.6f ICIR %.6f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q)))

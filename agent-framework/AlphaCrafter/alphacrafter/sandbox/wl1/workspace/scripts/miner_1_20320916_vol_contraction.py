import os,numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];root='../persistent/stock_data';px={}
for s in U:
 f=os.path.join(root,s+'.csv')
 if os.path.exists(f):
  d=pd.read_csv(f);d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close.astype(float)
p=pd.DataFrame(px).sort_index();r=p.pct_change();v20=r.rolling(20,min_periods=15).std();v60=r.rolling(60,min_periods=40).std();mom=p.pct_change(60);f=(v60/(v20+1e-12)-1).where(mom>0).shift(1); fy=p.shift(-10)/p-1
rows=[]
for dt in f.index:
 z=pd.DataFrame({'x':f.loc[dt],'y':fy.loc[dt]}).dropna()
 if len(z)>=8:rows.append((dt,z.x.corr(z.y),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('dates',len(r),'avg_n',r.n.mean(),'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
for a,b in [('2024','2026'),('2027','2029'),('2030','2032')]:
 q=r.loc[a:b];print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean())
for h in [5,20]:
 rr=[];ff=p.shift(-h)/p-1
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':ff.loc[dt]}).dropna()
  if len(z)>=8:rr.append(z.x.corr(z.y))
 print('h',h,'n',len(rr),'IC',np.nanmean(rr),'ICIR',np.nanmean(rr)/np.nanstd(rr,ddof=1))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20320916_vol_contraction_signal.csv',index=False)

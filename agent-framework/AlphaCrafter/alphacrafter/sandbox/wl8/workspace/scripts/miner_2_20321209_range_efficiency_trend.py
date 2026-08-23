import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
  D[s]=x['close'].astype(float).replace(0,np.nan)
 except FileNotFoundError: pass
p=pd.DataFrame(D).sort_index(); lr=np.log(p).diff()
# Range-efficiency trend: directional displacement relative to total path length.
net=np.log(p/p.shift(20)); path=lr.abs().rolling(20,min_periods=15).sum(); raw=net/path.replace(0,np.nan)
# lag one session and smooth, preserving interpretable bounded trend persistence
f=raw.shift(1).rolling(3,min_periods=2).mean()
q=lambda h: np.log(p.shift(-h)/p)
def ics(h):
 out=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q(h).loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): out.append((d,c,len(z)))
 return out
rows=ics(10); ser=pd.Series([x[1] for x in rows],index=[x[0] for x in rows])
print('dates',len(ser),'avgN',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15)
print('IC %.6f ICIR %.6f hit %.4f'%(ser.mean(),ser.mean()/ser.std(ddof=1),(ser>0).mean()))
for n in [365,750,1260]:
 z=ser.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 a=[x[1] for x in ics(h)]; print('decay',h,'%.6f'%np.nanmean(a))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20321209_range_efficiency_trend_signal.csv',index=False)
pd.DataFrame({'date':ser.index,'ic':ser}).to_csv('scripts/miner_2_20321209_range_efficiency_trend_ic.csv',index=False)

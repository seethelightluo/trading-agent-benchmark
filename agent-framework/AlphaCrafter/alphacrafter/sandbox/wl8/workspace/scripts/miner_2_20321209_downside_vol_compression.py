import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); D[s]=x.close.astype(float).replace(0,np.nan)
 except FileNotFoundError: pass
p=pd.DataFrame(D).sort_index(); r=np.log(p).diff()
# Volatility contraction: lagged short downside RMS relative to long downside RMS; low ratio is preferred.
down=r.where(r<0,0.0); s20=np.sqrt((down**2).rolling(20,min_periods=15).mean()); s60=np.sqrt((down**2).rolling(60,min_periods=40).mean())
f=-(s20/s60.replace(0,np.nan)).shift(1).rolling(3,min_periods=2).mean()
def calc(h):
 q=np.log(p.shift(-h)/p); out=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):out.append((d,c,len(z)))
 return out
rows=calc(10); ic=pd.Series([x[1] for x in rows],index=[x[0] for x in rows]); print('dates',len(ic),'avgN',np.mean([x[2] for x in rows]),'coverage',np.mean([x[2] for x in rows])/15); print('IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1),(ic>0).mean()));
for n in [365,750,1260]:
 z=ic.tail(n);print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:print('decay',h,np.nanmean([x[1] for x in calc(h)]))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20321209_downside_vol_compression_signal.csv',index=False);pd.DataFrame({'date':ic.index,'ic':ic}).to_csv('scripts/miner_2_20321209_downside_vol_compression_ic.csv',index=False)

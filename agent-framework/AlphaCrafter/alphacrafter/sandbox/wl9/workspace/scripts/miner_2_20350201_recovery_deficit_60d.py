import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>200}).sort_index()
r=px.pct_change(); lo=px.rolling(60,min_periods=40).min(); vol=r.rolling(60,min_periods=40).std()
# Contrarian recovery deficit: assets still close to their trailing low receive high scores; volatility scaled.
f=(-(px/(lo+1e-12)-1)/(vol+1e-8)).shift(1).replace([np.inf,-np.inf],np.nan)
f.to_csv('scripts/miner_2_20350201_recovery_deficit_60d_signal.csv',index_label='date')
print('panel_dates=%d instruments=%d' %(len(px),len(U)))
for h in [10,20,40,60,80]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c); ns.append(len(z))
 x=pd.Series(vals); print('H=%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f' %(h,len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252),(x>0).mean()))
print('coverage=%.6f' %(f.notna().sum(axis=1).mean()/len(U)))
fr=px.shift(-60)/px-1
for name,a,b in [('2020-23','2020','2023'),('2024-26','2024','2026'),('2027-29','2027','2029'),('2030-32','2030','2032'),('2033-35','2033','2035')]:
 vals=[]
 for dt in f.index:
  if a<=str(dt)[:4]<=b:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(c): vals.append(c)
 x=pd.Series(vals); print('regime=%s dates=%d IC=%s ICIR=%s hit=%s' %(name,len(x),'%.6f'%x.mean() if len(x) else 'nan','%.6f'%(x.mean()/x.std(ddof=1)*np.sqrt(252)) if len(x)>1 else 'nan','%.4f'%((x>0).mean()) if len(x) else 'nan'))

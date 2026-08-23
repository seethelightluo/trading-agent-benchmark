import numpy as np,pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-04-28'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:CUT]
r=P.pct_change(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill()
ret20=P/P.shift(20)-1; neg=r.clip(upper=0); down=np.sqrt((neg**2).rolling(40,min_periods=10).mean())*np.sqrt(20); stress=(vix/vix.rolling(60,min_periods=40).median()).clip(.5,2)
f=(-ret20/(down+0.005)).mul(1+0.5*(stress-1).clip(lower=0),axis=0)
print('cutoff',CUT.date(),'dates',len(P),'assets',len(U),'coverage',round(f.notna().stack().mean(),4),flush=True)
for h in [5,10,20]:
 z=[];ns=[];ds=[]
 for i in range(len(P)-h):
  a=pd.concat([f.iloc[i].rename('x'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(a)>=8:
   q=spearmanr(a.x,a.y).statistic
   if pd.notna(q):z.append(q);ns.append(len(a));ds.append(P.index[i])
 z=np.array(z); print('h=%d dates=%d avg_n=%.2f cov=%.4f IC=%+.8f ICIR=%+.6f hit=%.4f'%(h,len(z),np.mean(ns),np.mean(ns)/15,z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),np.mean(z>0)),flush=True)
 for a,b in [(2026,2028),(2029,2030),(2031,2032)]:
  q=[z[j] for j,d in enumerate(ds) if a<=d.year<=b]; print(' regime',a,b,'n',len(q),'ic',round(float(np.mean(q)),6) if q else None)
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),flush=True)

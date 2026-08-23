import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2033-01-06')
D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.astype(float).replace(0,np.nan)
 except FileNotFoundError: pass
p=pd.DataFrame(D).sort_index(); p=p.loc[p.index<=cutoff]; r=np.log(p).diff()
f=(-r.rolling(40,min_periods=30).sum()/r.rolling(40,min_periods=30).std().replace(0,np.nan)).shift(1).rolling(3,min_periods=2).mean()
def calc(h):
 q=np.log(p.shift(-h)/p); out=[]
 for d in f.index:
  z=pd.concat([f.loc[d],q.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): out.append((d,c,len(z)))
 return out
for h in [1,5,10,20]:
 a=calc(h); x=pd.Series([v[1] for v in a],index=[v[0] for v in a]); print('horizon',h,'dates',len(x),'avgN',np.mean([v[2] for v in a]),'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
a=calc(10); x=pd.Series([v[1] for v in a],index=[v[0] for v in a]);
for n in [365,750,1260]:
 z=x.tail(n); print('recent',n,'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
print('turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'last_validation',x.index[-1].date())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20330106_invvolnorm40_signal.csv',index=False)
pd.DataFrame({'date':x.index,'ic':x}).to_csv('scripts/miner_2_20330106_invvolnorm40_ic.csv',index=False)

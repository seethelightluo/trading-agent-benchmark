import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-02-18')
def load(s):
 return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'],usecols=['date','close']).set_index('date').close.rename(s)
px=pd.concat([load(s) for s in U],axis=1).sort_index().loc[:END]
r=px.pct_change(); w=60
# Distance from the trailing drawdown midpoint: 0 at rolling low, 1 at rolling high.
hi=px.rolling(w,min_periods=w).max(); lo=px.rolling(w,min_periods=w).min()
progress=((px-lo)/(hi-lo).replace(0,np.nan)).clip(0,1)
# Prefer recent losers that have already recovered materially from their trough.
f=(-r.rolling(10,min_periods=10).sum())*(0.7+0.6*progress)
def csic(a,b):
 ok=a.notna()&b.notna(); n=ok.sum(axis=1); ar=a.rank(axis=1); br=b.rank(axis=1)
 am=ar.where(ok).mean(axis=1); bm=br.where(ok).mean(axis=1)
 ac=ar.sub(am,axis=0); bc=br.sub(bm,axis=0)
 z=(ac*bc).sum(axis=1)/np.sqrt((ac*ac).sum(axis=1)*(bc*bc).sum(axis=1)); return z[(n>=8)&z.notna()],n
print('DATES',px.index.min().date(),px.index.max().date(),'ASSETS',len(U))
for h in [1,5,10,20]:
 z,n=csic(f,px.pct_change(h).shift(-h)); print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),len(z),n.loc[z.index].mean(),(z>0).mean()))
 if h==10:
  for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034'),('2035','2035')]:
   q=z.loc[a:b]; print('REG',a,b,'IC %.6f ICIR %.4f N %d hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),len(q),(q>0).mean()))
print('COVER %.4f TURN %.4f'%(f.notna().mean().mean(),f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean()))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_2_20350219_recovery_progress_reversal_signal.csv',index=False)

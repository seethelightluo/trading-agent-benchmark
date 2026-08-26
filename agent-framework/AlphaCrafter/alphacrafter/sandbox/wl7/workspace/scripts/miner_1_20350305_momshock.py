import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-03-04')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'],usecols=['date','close']).set_index('date').close
px=pd.concat([load(s).rename(s) for s in U],axis=1).sort_index().loc[:END]; r=px.pct_change()
# Candidate: medium-horizon momentum, normalized by realized risk and penalized by recent shockiness.
mom=r.rolling(20,min_periods=20).sum(); vol=r.rolling(40,min_periods=30).std(); shock=r.abs().rolling(10,min_periods=10).mean(); base=r.abs().rolling(40,min_periods=30).mean()
f=mom/(vol*np.sqrt(20)+1e-12) * (1/(1+(shock/(base+1e-12)-1).clip(lower=0)))
def ic(a,b):
 ok=a.notna()&b.notna(); n=ok.sum(axis=1); ar=a.rank(axis=1); br=b.rank(axis=1); am=ar.where(ok).mean(axis=1); bm=br.where(ok).mean(axis=1); x=(ar-am.values[:,None]).where(ok); y=(br-bm.values[:,None]).where(ok); return ((x*y).sum(axis=1)/np.sqrt((x*x).sum(axis=1)*(y*y).sum(axis=1))),n
print('DATES',px.index.min().date(),px.index.max().date(),'ASSETS',px.shape[1])
for h in [1,5,10,20]:
 z,n=ic(f,px.pct_change(h).shift(-h)); z=z[(n>=8)&z.notna()]; print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),len(z),n.loc[z.index].mean(),(z>0).mean()))
 if h==10:
  for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034'),('2035','2035')]:
   q=z.loc[a:b]; print('REG',a,b,'IC %.6f ICIR %.4f N %d hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),len(q), (q>0).mean()))
print('COVER',f.notna().mean().mean(),'TURN',f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_1_20350305_momshock_signal.csv',index=False)

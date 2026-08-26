import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-02-04')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'],usecols=['date','close']).set_index('date')
df=pd.concat([load(s).close.rename(s) for s in U],axis=1,sort=True).sort_index().loc[:END]
r=df.pct_change(10); peak=df.rolling(60,min_periods=40).max(); below=df < peak*(1-1e-12)
def run_duration(x):
 out=[]; k=0
 for v in x:
  k=k+1 if bool(v) else 0; out.append(min(k,60))
 return pd.Series(out,index=x.index)
dur=below.apply(run_duration)
f=(-r)*(1+0.4*(dur/60.0))
def ic(a,b):
 ok=a.notna()&b.notna(); n=ok.sum(axis=1); ar=a.rank(axis=1); br=b.rank(axis=1)
 am=ar.where(ok).mean(axis=1); bm=br.where(ok).mean(axis=1)
 x=(ar-am.values[:,None]).where(ok); y=(br-bm.values[:,None]).where(ok)
 return (x*y).sum(axis=1)/np.sqrt((x*x).sum(axis=1)*(y*y).sum(axis=1)),n
print('DATES',df.index.min().date(),df.index.max().date(),'ASSETS',df.shape[1])
for h in [1,5,10,20]:
 z,n=ic(f,df.pct_change(h).shift(-h)); z=z[(n>=8)&z.notna()]
 print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(252),len(z),n.loc[z.index].mean(),(z>0).mean()))
 if h==10:
  for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034'),('2035','2035')]:
   q=z.loc[a:b]; print('REG',a,b,'IC %.6f ICIR %.4f N %d hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),len(q),(q>0).mean()))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); out.to_csv('scripts/miner_2_20350205_drawdown_duration_reversal10_signal.csv',index=False)
print('COVER',f.notna().mean().mean(),'TURN',f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())

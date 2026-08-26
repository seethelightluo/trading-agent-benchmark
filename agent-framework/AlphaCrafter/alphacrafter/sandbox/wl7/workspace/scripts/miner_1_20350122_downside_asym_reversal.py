import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-01-20')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'],usecols=['date','close']).set_index('date').close
px=pd.concat([load(s).rename(s) for s in U],axis=1).sort_index().loc[:END]; r=px.pct_change()
# Downside-asymmetry reversal: recent return, signed by the fraction of path volatility
# contributed by losses; high negative-return dominance makes a rebound more likely.
neg=r.clip(upper=0)
down=neg.pow(2).rolling(20,min_periods=15).mean().pow(.5)
total=r.pow(2).rolling(20,min_periods=15).mean().pow(.5)
asym=down.div(total.replace(0,np.nan))
f=(-r.rolling(10,min_periods=10).sum()).div((1+asym).replace(0,np.nan))
def ic(a,b):
 ok=a.notna()&b.notna(); n=ok.sum(axis=1); ar=a.rank(axis=1); br=b.rank(axis=1)
 am=ar.where(ok).mean(axis=1); bm=br.where(ok).mean(axis=1)
 x=(ar-am.values[:,None]).where(ok); y=(br-bm.values[:,None]).where(ok)
 return (x*y).sum(axis=1)/np.sqrt((x*x).sum(axis=1)*(y*y).sum(axis=1)),n
print('DATES',px.index.min().date(),px.index.max().date(),'ASSETS',px.shape[1])
for h in [1,5,10,20]:
 z,n=ic(f,px.pct_change(h).shift(-h)); z=z[(n>=8)&z.notna()]
 print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1)*np.sqrt(len(z)),len(z),n.loc[z.index].mean(),(z>0).mean()))
 if h==10:
  for a,b in [('2023','2026'),('2027','2030'),('2031','2035')]:
   q=z.loc[a:b]; print('REG',a,b,'IC %.6f ICIR %.4f N %d hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),len(q),(q>0).mean()))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal'); out.to_csv('scripts/miner_1_20350122_downside_asym_reversal_signal.csv',index=False)
print('COVER',f.notna().mean().mean(),'TURN',f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())

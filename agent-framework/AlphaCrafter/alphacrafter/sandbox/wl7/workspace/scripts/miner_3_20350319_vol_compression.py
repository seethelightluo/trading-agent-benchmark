import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-03-18')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'],usecols=['date','close']).set_index('date')
px=pd.concat([load(s).close.rename(s) for s in U],axis=1).sort_index().loc[:END]; r=px.pct_change()
# Volatility compression: prefer assets with low recent realized volatility,
# normalized cross-sectionally to avoid scale differences.
vol=r.rolling(20,min_periods=20).std(); longvol=r.rolling(60,min_periods=60).std(); f=-(vol/longvol.replace(0,np.nan))
def ic(a,b):
 ok=a.notna()&b.notna(); n=ok.sum(1); ar=a.rank(1); br=b.rank(1); ac=ar-ar.where(ok).mean(1).values[:,None]; bc=br-br.where(ok).mean(1).values[:,None]; q=(ac*bc).sum(1)/np.sqrt((ac*ac).sum(1)*(bc*bc).sum(1)); return q[(n>=8)&q.notna()],n
print('DATES',px.index.min().date(),px.index.max().date(),'ASSETS',len(U))
for h in [1,5,10,20]:
 q,n=ic(f,px.pct_change(h).shift(-h)); print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),len(q),n.loc[q.index].mean(),(q>0).mean()))
 if h==10:
  for a,b in [('2023','2026'),('2027','2030'),('2031','2034'),('2035','2035')]:
   x=q.loc[a:b]; print('REG',a,b,'IC %.6f ICIR %.4f N %d'%(x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252),len(x)))
print('COVER',f.notna().mean().mean(),'TURN',f.rank(pct=True).diff().abs().mean(1).dropna().mean())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_3_20350319_vol_compression_signal.csv',index=False)

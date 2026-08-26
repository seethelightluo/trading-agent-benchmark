import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-04-15')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'],usecols=['date','close']).set_index('date')
px=pd.concat([load(s).close.rename(s) for s in U],axis=1).sort_index().loc[:END]
r=px.pct_change()
# Trend-exhaustion rebound: negative recent shock is rewarded only when the
# asset remains above its long-term floor; scale by downside volatility. This
# avoids buying structural collapses while exploiting short-horizon overshoot.
ret3=r.rolling(3,min_periods=3).sum()
ret20=r.rolling(20,min_periods=20).sum()
lo120=px.rolling(120,min_periods=120).min()
floor=(px/lo120-1).clip(0,3)
downvol=r.where(r<0).rolling(20,min_periods=10).std()
raw=(-ret3)*(0.5+floor)/(downvol.replace(0,np.nan)*np.sqrt(20))
# require the medium trend not to be deeply negative, but retain all valid names
f=raw.where(ret20>-0.35)
f=f.sub(f.median(axis=1),axis=0)
def csic(a,b):
 ok=a.notna()&b.notna(); n=ok.sum(axis=1); ar=a.rank(axis=1); br=b.rank(axis=1)
 ac=ar-ar.where(ok).mean(axis=1).values[:,None]; bc=br-br.where(ok).mean(axis=1).values[:,None]
 q=(ac*bc).sum(axis=1)/np.sqrt((ac*ac).sum(axis=1)*(bc*bc).sum(axis=1)); return q[(n>=8)&q.notna()],n
print('DATES',px.index.min().date(),px.index.max().date(),'ASSETS',len(U))
for h in [1,5,10,20]:
 q,n=csic(f,px.pct_change(h).shift(-h)); print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252),len(q),n.loc[q.index].mean(),(q>0).mean()))
 if h==10:
  for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034'),('2035','2035')]:
   x=q.loc[a:b]; print('REG',a,b,'IC %.6f ICIR %.4f N %d hit %.3f'%(x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(252),len(x), (x>0).mean()))
print('COVER',f.notna().mean().mean(),'TURN',f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_3_20350416_exhaustion_rebound_signal.csv',index=False)

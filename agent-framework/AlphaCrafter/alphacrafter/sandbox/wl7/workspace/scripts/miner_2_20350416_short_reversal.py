import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END=pd.Timestamp('2035-04-15')
def L(s):return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'],usecols=['date','close']).set_index('date').close
px=pd.concat([L(s).rename(s) for s in U],axis=1).sort_index().loc[:END];r=px.pct_change(); f=-r.rolling(5,min_periods=5).sum()/r.rolling(20,min_periods=20).std().replace(0,np.nan)
def C(a,b):
 ok=a.notna()&b.notna();n=ok.sum(1);x=a.rank(1);y=b.rank(1);x=x-x.where(ok).mean(1).values[:,None];y=y-y.where(ok).mean(1).values[:,None];q=(x*y).sum(1)/np.sqrt((x*x).sum(1)*(y*y).sum(1));return q[(n>=8)&q.notna()],n
for h in [1,5,10,20]:
 q,n=C(f,px.pct_change(h).shift(-h));print('H',h,'IC %.6f ICIR %.4f N %d avgN %.2f hit %.3f'%(q.mean(),q.mean()/q.std()*np.sqrt(252),len(q),n[q.index].mean(),(q>0).mean()))
print('COVER',f.notna().mean().mean(),'TURN',f.rank(pct=True).diff().abs().mean(1).dropna().mean())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_2_20350416_short_reversal_signal.csv',index=False)

import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-02-06')
C={}; H={}; L={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date')
 x=x[x.date<=END].set_index('date'); C[s]=x.close; H[s]=x.high; L[s]=x.low
px=pd.DataFrame(C).sort_index(); hi=pd.DataFrame(H).reindex(px.index); lo=pd.DataFrame(L).reindex(px.index)
prev=px.shift(1)
a=(hi-lo)/prev; b=(hi-prev).abs()/prev; c=(lo-prev).abs()/prev
tr=pd.concat([a,b,c],keys=['a','b','c'],axis=1).T.groupby(level=1).max().T
atr=tr.rolling(20,min_periods=15).mean().shift(1)
sig=(px.pct_change(20).shift(1)/atr).replace([np.inf,-np.inf],np.nan).clip(-10,10)
for h in [1,3,5,10]:
 f=px.shift(-h)/px-1; rows=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN %.2f'%z.n.mean(),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('coverage',sig.notna().sum().sum()/sig.size)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20290208_range_adjusted_momentum_signal.csv',index=False)

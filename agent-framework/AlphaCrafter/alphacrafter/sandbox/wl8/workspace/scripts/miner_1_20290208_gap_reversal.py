import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-02-06'); O={};C={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date');x=x[x.date<=END].set_index('date');O[s]=x.open;C[s]=x.close
op=pd.DataFrame(O).sort_index();cl=pd.DataFrame(C).reindex(op.index)
# Reversal of the lagged overnight gap, smoothed over 3 completed sessions
sig=-(op/cl.shift(1)-1).rolling(3,min_periods=2).mean().shift(1)
for h in [1,3,5,10]:
 f=cl.shift(-h)/cl-1; rows=[]
 for d in cl.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN %.2f'%z.n.mean(),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('coverage',sig.notna().sum().sum()/sig.size)
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20290208_gap_reversal_signal.csv',index=False)

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-02-20')
C={}; V={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); x=x[x.date<=END].set_index('date'); C[s]=x.close; V[s]=x.volume
cl=pd.DataFrame(C).sort_index(); vol=pd.DataFrame(V).reindex(cl.index)
# Volatility-scaled short-term reversal, all inputs lagged one completed session.
r=cl.pct_change(5); rv=cl.pct_change().rolling(20,min_periods=15).std().shift(1)
sig=(-r.div(rv).clip(-8,8)).shift(1)
# volume surprise is deliberately not used in signal; retain broad coverage and interpretable reversal
for h in [1,3,5,10]:
 f=cl.shift(-h)/cl-1; rows=[]
 for d in cl.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN %.2f'%z.n.mean(),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('coverage',sig.notna().sum().sum()/sig.size,'dates',len(cl),'assets',len(U))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20290222_short_reversal_signal.csv',index=False)

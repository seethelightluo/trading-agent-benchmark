import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-03-06')
C={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').set_index('date').close for s in U}
px=pd.DataFrame(C).sort_index(); r=px.pct_change()
# Trend efficiency: signed 20d displacement divided by total absolute path, lagged one day.
path=r.abs().rolling(20,min_periods=15).sum(); sig=(px.pct_change(20)/path).shift(1).clip(-1,1)
for h in [1,3,5,10]:
 f=px.shift(-h)/px-1; rows=[]
 for d in px.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]},index=U).dropna()
  if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1: rows.append((d,spearmanr(g.s,g.f).statistic,len(g)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 for lab,q in [('full',z),('recent180',z.tail(180)),('recent360',z.tail(360))]:
  print('h',h,lab,'dates',len(q),'avgN %.2f'%q.n.mean(),'IC %.6f ICIR %.6f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
print('coverage',sig.notna().sum().sum()/sig.size,'period',z.index.min().date(),z.index.max().date())
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20290308_efficiency_trend_signal.csv',index=False)

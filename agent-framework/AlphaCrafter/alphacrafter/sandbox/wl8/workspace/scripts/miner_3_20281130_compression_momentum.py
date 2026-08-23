import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-11-14')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']); P[s]=x[x.date<=END].set_index('date').close.sort_index()
p=pd.DataFrame(P).sort_index(); r=p.pct_change()
ret10=p.pct_change(10); vol20=r.rolling(20).std(); vol60=r.rolling(60).std()
compression=(vol60/vol20).clip(.5,2); f=(ret10/vol20*compression).shift(1).clip(-8,8)
for h in [1,3,5,10]:
 fr=p.shift(-h)/p-1; rows=[]
 for d in f.index:
  z=pd.DataFrame({'s':f.loc[d],'f':fr.loc[d]}).dropna()
  if len(z)>=8 and z.s.nunique()>1 and z.f.nunique()>1: rows.append((d,spearmanr(z.s,z.f).statistic,len(z)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z[z.index>=END-pd.Timedelta(days=180)]
 print('h',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC %.6f ICIR %.6f hit %.4f recentIC %.6f recentICIR %.6f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean(),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('assets',len(p.columns),'rows',len(p),'coverage',round(f.notna().mean().mean(),4),'nonzero',round((f!=0).mean().mean(),4))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20281130_compression_momentum_signal.csv',index=False)

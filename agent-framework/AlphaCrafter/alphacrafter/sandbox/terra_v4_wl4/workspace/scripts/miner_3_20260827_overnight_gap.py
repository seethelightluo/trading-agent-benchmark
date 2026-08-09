import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-07-15')
def ld(s):
 p='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return x[['open','close']]
d={s:ld(s) for s in U}
# completed-session overnight gap, using prior completed close and today's completed open
f=pd.concat([(-(x.open/x.close.shift(1)-1)).rename(s) for s,x in d.items()],axis=1)
cl=pd.concat([x.close.rename(s) for s,x in d.items()],axis=1)

def ev(h):
 fw=cl.shift(-h)/cl-1; out=[]
 for dt in f.index[f.index<=C]:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); return q
for h in [1,5,10]:
 q=ev(h); print('H',h,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean())
q=ev(1)
for y in range(2020,2027):
 z=q[q.index.year==y];
 if len(z): print('YEAR',y,'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'dates',len(z))
print('coverage',f.loc[:C].notna().sum().sum()/(len(f.loc[:C])*15),'turnover',f.loc[:C].rank(pct=True).diff().abs().mean().mean())
f.loc[:C].to_csv('scripts/miner_3_20260827_overnight_gap_signal.csv')

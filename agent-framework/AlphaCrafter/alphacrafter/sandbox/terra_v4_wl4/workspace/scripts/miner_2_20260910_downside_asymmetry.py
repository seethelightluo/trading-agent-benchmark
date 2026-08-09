import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-07-15')
def load(s):
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return x.close
cl=pd.concat([load(s).rename(s) for s in U],axis=1,sort=True); r=cl.pct_change()
for w in [20,40]:
 fac=pd.DataFrame(index=cl.index)
 for s in U:
  rs=r[s].dropna(); dn=rs.where(rs<0).rolling(w,min_periods=max(10,w//2)).std(); tv=rs.rolling(w,min_periods=max(10,w//2)).std(); fac[s]=(-dn/tv).reindex(cl.index)
 fac.loc[:C].to_csv(f'scripts/miner_2_20260910_downside_asymmetry_{w}d_signal.csv')
 results={}
 for h in [1,5,10]:
  fw=cl.shift(-h)/cl-1; out=[]
  for dt in fac.index[fac.index<=C]:
   z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
   if len(z)>=8: out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  q=pd.DataFrame(out,columns=['date','ic','n']); q['date']=pd.to_datetime(q.date);q=q.set_index('date');results[h]=q
  print('W',w,'H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(),6),'hit',round((q.ic>0).mean(),4))
 if w==20:
  q=results[1]
  for y in range(2020,2027):
   a=q[q.index.year==y]
   if len(a): print('YEAR',y,'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(),6),'dates',len(a))
  print('coverage',round(fac.loc[:C].notna().sum().sum()/(len(fac.loc[:C])*15),4),'turnover',round(fac.loc[:C].rank(pct=True).diff().abs().mean().mean(),4))

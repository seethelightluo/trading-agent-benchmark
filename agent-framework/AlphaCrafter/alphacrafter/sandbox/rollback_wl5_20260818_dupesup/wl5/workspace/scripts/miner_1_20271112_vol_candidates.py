import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 try:D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:pd.Timestamp('2027-11-12')].close
 except:pass
px=pd.DataFrame(D); r=px.pct_change();
def test(name,f):
 fw=px.shift(-10)/px-1; rows=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=np.array([x[1] for x in rows]);print(name,len(a),np.mean([x[2] for x in rows])/15,a.mean(),a.std(ddof=1),a.mean()/a.std(ddof=1),np.mean(a>0), 'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-11-12')]:
  q=[v for d,v,n in rows if pd.Timestamp(lo)<=d<=pd.Timestamp(hi)];print(lo,len(q),np.mean(q) if q else None)
test('vol_compression_20',-r.rolling(20).std())
test('vol_compression_10',-r.rolling(10).std())
test('range_eff_10',(pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:pd.Timestamp('2027-11-12')].high for s in U if s in D})-pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:pd.Timestamp('2027-11-12')].low for s in U if s in D})).rolling(10).mean()/(r.abs().rolling(10).sum()+1e-9))

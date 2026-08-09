import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-07-15')
def ld(s,p='../persistent/stock_data/'): return pd.read_csv(p+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].sort_index()
px=pd.concat([ld(s).rename(s) for s in U]+[ld('DXY','../persistent/index_data/').rename('DXY')],axis=1,join='inner'); r=px.pct_change(); v=r.DXY.rolling(120,min_periods=80).var(); b20=r[U].rolling(20,min_periods=15).cov(r.DXY).div(r.DXY.rolling(20,min_periods=15).var(),axis=0); b120=r[U].rolling(120,min_periods=80).cov(r.DXY).div(v,axis=0); f=-(b20-b120)
def ev(h):
 fw=pd.concat([(px[s].shift(-h)/px[s]-1).rename(s) for s in U],axis=1); out=[]
 for d in f.index[f.index<=C]:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
for h in [1,5,10]:
 q=ev(h);print(h,len(q),q.n.mean(),q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean())
for y in [(2020,2022),(2023,2024),(2025,2026)]:
 q=ev(1);q=q[(q.index.year>=y[0])&(q.index.year<=y[1])];print(y,q.ic.mean(),q.ic.mean()/q.ic.std(),len(q))
print('turn',f.loc[:C].rank(axis=1,pct=True).diff().abs().mean().mean())
f.loc[:C].to_csv('scripts/miner_3_20260730_dxy_beta_change_signal.csv')

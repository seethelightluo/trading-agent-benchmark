import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'] for s in U}).sort_index()
r=p.pct_change(); vx=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].reindex(p.index).ffill().pct_change()
shock=vx.gt(0); f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i,dt in enumerate(p.index):
 if i<60: continue
 h=r.iloc[i-60:i]; sh=shock.iloc[i-60:i]
 if sh.sum()<5: continue
 f.loc[dt]=-(h.where(sh,axis=0).mean()-h.mean())

def run(horizon):
 fw=p.pct_change(horizon).shift(-horizon); rows=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1: rows.append((dt,spearmanr(a.f,a.y).statistic,len(a)))
 z=np.array([x[1] for x in rows]); print('h',horizon,'dates',len(z),'meanN',np.mean([x[2] for x in rows]),'IC %.8f ICIR %.8f hit %.6f std %.8f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean(),z.std(ddof=1)))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2028)]:
  q=np.array([x[1] for x in rows if lo<=x[0].year<=hi]); print('regime',lo,hi,'mean',q.mean() if len(q) else np.nan,'n',len(q))
 return rows
run(5);run(1);run(10)
print('coverage %.6f turnover %.6f instruments %d period %s %s'%(f.notna().sum().sum()/f.size,f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),len(U),p.index.min().date(),p.index.max().date()))

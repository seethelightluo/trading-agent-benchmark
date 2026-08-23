import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2028-01-12')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date')
d={s:load(s) for s in U}; c=pd.concat({s:x.close for s,x in d.items()},axis=1).sort_index(); r=c.pct_change()
# Reversal amplified for assets whose recent risk was downside-skewed: fade 10d return / 20d downside deviation.
down=r.clip(upper=0).rolling(20,min_periods=15).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
raw=r.rolling(10,min_periods=8).sum()/(down*np.sqrt(20)+1e-8); f=-raw.sub(raw.median(axis=1),axis=0)
def eval(y):
 rows=[]
 for dt in f.index:
  z=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1: rows.append((dt,len(z),spearmanr(z.f,z.y).statistic))
 return pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
a=eval(c.pct_change(10).shift(-10)); print('dates',len(a),'avgN',round(a.n.mean(),2),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for lo,hi in [(2020,2021),(2022,2023),(2024,2025),(2026,2027),(2028,2028)]:
 q=a[(a.index.year>=lo)&(a.index.year<=hi)].ic; print('reg',lo,hi,'n',len(q),'ic',round(q.mean(),6) if len(q) else None,'ir',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)
for h in [1,5,20]:
 b=eval(c.pct_change(h).shift(-h)); print('horizon',h,'dates',len(b),'IC',round(b.ic.mean(),6),'ICIR',round(b.ic.mean()/b.ic.std(ddof=1),6))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_1_20280113_downside_adjusted_reversal_signal.csv',index=False)

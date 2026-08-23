import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-09-23'); prices={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index(); prices[s]=d
P=pd.concat(prices,axis=1).sort_index(); P=P.loc[P.index<=end]
r=P.pct_change(); spx20=P['SPX']/P['SPX'].shift(20)-1
f=(P/P.shift(20)-1).apply(lambda c: c-spx20)
f=f/(r.rolling(20).std()*np.sqrt(20))
def calc(h):
 rows=[]
 for i,dt in enumerate(P.index[:-h]):
  vals=f.loc[dt]; fut=P.iloc[i+h]/P.iloc[i]-1; ok=vals.notna()&fut.notna()
  if ok.sum()>=8: rows.append((dt,spearmanr(vals[ok],fut[ok]).statistic,ok.sum()))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def stat(x): return len(x),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean(),x.n.mean()
for h in [1,5,10]:
 x=calc(h); print('horizon',h,'dates',len(x),'assets',x.n.mean(),'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
 for lab, y in [('2020-22',x.loc['2020':'2022']),('2023-24',x.loc['2023':'2024']),('2025-26',x.loc['2025':'2026'])]: print(lab,stat(y) if len(y) else 'none')
ranks=f.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean(),'coverage',calc(1).n.mean()/15)

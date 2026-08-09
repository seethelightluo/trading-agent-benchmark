import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-07-15')
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cutoff').set_index('date')['close'].reindex(r.index).ffill().pct_change()
rows={1:[],5:[],10:[]}; fs=[]
for i in range(70,len(r)-10):
 hist=r.iloc[i-60:i]; shock=v.iloc[i-60:i]>0
 if shock.sum()<5: continue
 f=hist.where(np.repeat(shock.values[:,None],len(assets),axis=1)).mean()-hist.mean(); fs.append((r.index[i],f))
 for h in rows:
  z=pd.concat([f,r.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(z)>=8: rows[h].append((r.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
for h,x in rows.items():
 ic=pd.Series(dict(x)); print('H',h,'N',len(ic),'mean instruments',r.loc[ic.index].notna().sum(axis=1).mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=ic[(ic.index.year>=lo)&(ic.index.year<=hi)]; print(lo,round(z.mean(),5),len(z))
q=pd.DataFrame({d:f for d,f in fs}).T.rank(axis=1,pct=True); print('turnover',q.diff().abs().mean().mean(),'coverage',r.notna().mean().mean(),'cutoff',r.index[-1])

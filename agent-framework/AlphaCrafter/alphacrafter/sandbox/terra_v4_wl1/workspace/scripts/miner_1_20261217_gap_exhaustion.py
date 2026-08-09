import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
p=pd.DataFrame({s:d.close for s,d in D.items()}).sort_index(); o=pd.DataFrame({s:d.open for s,d in D.items()}).reindex(p.index)
r=p.pct_change(); gap=o/p.shift(1)-1
# interpretable gap-exhaustion reversal: fade the recent mean overnight gap
for w in [1,3,5]:
 f=-gap.rolling(w,min_periods=w).mean(); rows=[]
 for i in range(len(p)-1):
  q=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8: rows.append((p.index[i],spearmanr(q.f,q.y).statistic,len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']); x=a.ic
 print('WINDOW',w,'DATES',len(x),'AVG_N',round(a.n.mean(),2),'COVERAGE',round(a.n.mean()/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'HIT',round((x>0).mean(),4),'TURN',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=x[(pd.to_datetime(a.date).dt.year>=lo)&(pd.to_datetime(a.date).dt.year<=hi)]; print('REGIME',lo,hi,'N',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
 print('DECAY',[(h, round(np.nanmean([spearmanr(pd.concat([f.iloc[i].rename('f'),(p.shift(-h).div(p)-1).iloc[i].rename('y')],axis=1).dropna().f, pd.concat([f.iloc[i].rename('f'),(p.shift(-h).div(p)-1).iloc[i].rename('y')],axis=1).dropna().y).statistic for i in range(len(p)-h) if len(pd.concat([f.iloc[i].rename('f'),(p.shift(-h).div(p)-1).iloc[i].rename('y')],axis=1).dropna())>=8]),6)) for h in [1,5,10]])

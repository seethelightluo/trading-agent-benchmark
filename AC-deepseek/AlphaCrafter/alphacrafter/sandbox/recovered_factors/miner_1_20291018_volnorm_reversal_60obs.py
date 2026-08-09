import pandas as pd,numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date'); p[a]=d.close
p=pd.DataFrame(p).sort_index(); r=p.pct_change()
# Contrarian medium-term return, scaled by recent realized risk, lagged one session.
f=(-(p.pct_change(60))/(r.rolling(40,min_periods=30).std()*np.sqrt(40))).shift(1)
print(f'rows={len(p)} assets={len(assets)} dates={p.index.min().date()}..{p.index.max().date()}')
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; vals=[]; ds=[]; ns=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic);ds.append(dt);ns.append(ok.sum())
 s=pd.Series(vals,index=ds); print(f'h={h} dates={len(s)} meanN={np.mean(ns):.2f} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={(s>0).mean():.4f}')
 for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-10-16')]:
  q=s.loc[lo:hi]; print(f' regime={lo} n={len(q)} IC={q.mean() if len(q) else np.nan:.6f} ICIR={q.mean()/q.std(ddof=1) if len(q)>1 else np.nan:.6f}')
rank=f.rank(axis=1,pct=True); print(f'coverage={f.notna().sum().sum()/f.size:.4f} turnover10={(rank-rank.shift(10)).abs().mean(axis=1).mean():.4f} mean_valid={f.notna().sum(axis=1).mean():.2f}')

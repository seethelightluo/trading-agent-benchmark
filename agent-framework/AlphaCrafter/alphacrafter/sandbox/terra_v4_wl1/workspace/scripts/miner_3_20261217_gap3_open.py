import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
C=pd.DataFrame({s:D[s].close for s in U}).sort_index(); O=pd.DataFrame({s:D[s].open for s in U}).reindex(C.index)
# prior-session close-to-open gap, smoothed over 3 sessions; reversal forecast next close-to-close return
G=O/C.shift(1)-1
F=-G.rolling(3,min_periods=3).mean()
Y=C.pct_change(1).shift(-1)
vals=[]; ns=[]; ds=[]
for d in C.index:
 q=pd.concat([F.loc[d].rename('f'),Y.loc[d].rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1:
  vals.append(spearmanr(q.f,q.y).statistic); ns.append(len(q)); ds.append(d)
a=np.asarray(vals); print('candidate gap3 reversal'); print('dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4)); print('coverage',round(F.notna().sum().sum()/F.size,4),'turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),6));
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-17')]:
 z=a[np.array([(d>=pd.Timestamp(lo) and d<=pd.Timestamp(hi)) for d in ds])]; print(lo,hi,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
# decay
for h in [1,5,10]:
 y=C.pct_change(h).shift(-h); x=[]
 for d in C.index:
  q=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:x.append(spearmanr(q.f,q.y).statistic)
 x=np.asarray(x);print('h',h,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
# artifact
out=pd.DataFrame(F.stack(),columns=['signal']);out.index.names=['date','symbol'];out.reset_index().to_csv('scripts/miner_3_20261217_gap3_open_signal.csv',index=False)

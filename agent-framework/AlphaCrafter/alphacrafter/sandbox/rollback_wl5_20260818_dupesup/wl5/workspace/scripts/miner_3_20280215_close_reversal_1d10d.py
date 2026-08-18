import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=4000) for s in U}; P=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in D.items()}).sort_index()
# One-day close-to-close relative reversal; forecast next 10 close-to-close sessions.
r=P.pct_change(); sig=-(r.sub(r.median(axis=1),axis=0)); fwd=P.shift(-10)/P.shift(-1)-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna(); print('dates',len(a),'avgN',a.n.mean(),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for x,y in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028')]:
 q=a.loc[x:y].ic
 if len(q):print(x,len(q),q.mean(),q.mean()/q.std(ddof=1))
rank=sig.rank(axis=1,pct=True); print('turnover',((rank-rank.shift()).abs().mean(axis=1)).mean(),'coverage',sig.notna().stack().mean())
out='scripts/miner_3_20280215_close_reversal_1d10d_signal.csv'; sig.stack().rename('signal').to_csv(out);print(out)

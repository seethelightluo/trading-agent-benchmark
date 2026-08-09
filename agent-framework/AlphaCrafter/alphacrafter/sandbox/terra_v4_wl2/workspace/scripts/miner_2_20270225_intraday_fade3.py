import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C={};O={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index(); C[a]=d.close;O[a]=d.open
p=pd.concat(C,axis=1).sort_index(); op=pd.concat(O,axis=1).reindex(p.index)
# 3-session mean intraday reversal, point-in-time
f=-(p/op-1).rolling(3).mean(); y=p.shift(-1)/p-1; rows=[]
for dt in f.index:
 ok=f.loc[dt].notna()&y.loc[dt].notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic,ok.sum()))
z=np.array([r[1] for r in rows]);print('dates',len(z),'avg_n',np.mean([r[2] for r in rows]),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
 q=np.array([r[1] for r in rows if str(r[0])[:10]>=lo and str(r[0])[:10]<=hi]);
 if len(q): print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').to_csv('../persistent/factor_signals_miner_2_20270225_intraday_fade3.csv',index=False)

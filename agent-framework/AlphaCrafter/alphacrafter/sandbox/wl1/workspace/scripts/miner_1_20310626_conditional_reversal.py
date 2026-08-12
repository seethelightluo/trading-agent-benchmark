import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index().ffill().loc[:'2031-06-26']; R=P.pct_change()
# short-horizon reversal conditioned on market breadth: fade yesterday's move, but only when breadth is extreme
breadth=(R.shift(1)>0).sum(axis=1)/15
F=R.shift(1).mul(-(1+0.8*((breadth-.5).abs())),axis=0)
rows=[]
for dt in F.index:
 a=F.loc[dt]; b=R.shift(-1).loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].rank(pct=True).corr(b[ok]),ok.sum()))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def st(x): return x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),len(x)
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.sum()/len(q)/15);print('daily',st(q.ic));print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-12-31'),('2029','2030-12-31'),('2031','2031-12-31')]: print(a,st(q.loc[a:b].ic))
F.to_csv('scripts/miner_1_20310626_conditional_reversal_signal.csv',index_label='date')

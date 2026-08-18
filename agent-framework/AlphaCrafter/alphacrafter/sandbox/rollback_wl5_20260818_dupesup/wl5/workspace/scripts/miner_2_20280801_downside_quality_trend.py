import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
base='../persistent/stock_data'; U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Downside deviation uses zeros for non-negative returns; all inputs lagged one session.
neg2=(r.clip(upper=0)**2).rolling(20,min_periods=20).mean().shift(1)
down=np.sqrt(neg2)
trend=P.pct_change(20).shift(1)/(down+1e-12)
shock=-P.pct_change(3).shift(1)/(r.rolling(20,min_periods=20).std().shift(1)+1e-12)
f=(trend+0.25*shock).replace([np.inf,-np.inf],np.nan)
fr=P.shift(-10)/P.shift(-1)-1; rows=[]
for dt in P.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
def st(q): return len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
print('candidate downside-quality trend + shock reversal universe',len(U),'dates',len(o),'meanN',round(o.n.mean(),3),'coverage',round(o.n.mean()/len(U),4))
print('IC ICIR hit',*[round(x,6) for x in st(o.ic)])
for a,b in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-12-31')]:
 q=o.loc[a:b].ic
 if len(q): print('regime',a,'n',len(q),'IC ICIR hit',*[round(x,6) for x in st(q)])
for k in [60,120,252]:
 q=o.tail(k).ic; print('recent',k,'IC ICIR hit',*[round(x,6) for x in st(q)])
rank=f.rank(axis=1,pct=True); rr=[]
for i in range(1,len(rank)):
 z=pd.concat([rank.iloc[i-1],rank.iloc[i]],axis=1).dropna()
 if len(z)>=8: rr.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('turnover_proxy',round(float(np.nanmean(rr)),6))
sig=f.stack().rename('signal').reset_index(); sig.columns=['date','symbol','signal']; sig.to_csv('scripts/miner_2_20280801_downside_quality_trend_signal.csv',index=False)
print('artifact scripts/miner_2_20280801_downside_quality_trend_signal.csv')

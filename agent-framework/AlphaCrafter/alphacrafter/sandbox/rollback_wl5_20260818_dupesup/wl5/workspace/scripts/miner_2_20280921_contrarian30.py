import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cutoff=pd.Timestamp('2028-09-21')
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d.date=pd.to_datetime(d.date); px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:cutoff]; r=P.pct_change(); vol=r.rolling(20,min_periods=15).std().shift(1)
# Contrarian risk-adjusted 30-session return, lagged one session; predict 10-session forward return.
f=(-P.pct_change(30).shift(1)/(vol*np.sqrt(30)+1e-12)).replace([np.inf,-np.inf],np.nan)
fr=P.shift(-10)/P-1; rows=[]
for dt in P.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
o=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); q=o.ic
def st(x): return len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean())
print('candidate contrarian_risk_adjusted_30d universe',len(U),'dates',len(o),'meanN',o.n.mean(),'coverage',o.n.mean()/len(U));print('overall n IC ICIR hit',st(q))
for name,x in [('2025-26',q.loc['2025-01-01':'2026-12-31']),('2027-28',q.loc['2027-01-01':]),('recent60',q.tail(60)),('recent120',q.tail(120)),('recent252',q.tail(252))]: print(name,st(x))
ranks=f.rank(axis=1,pct=True); turn=(ranks-ranks.shift(1)).abs().mean(axis=1); print('turnover_proxy',float(turn.loc[o.index].mean()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20280921_contrarian30_signal.csv',index=False);o.to_csv('scripts/miner_2_20280921_contrarian30_ic.csv')
print('artifacts written')

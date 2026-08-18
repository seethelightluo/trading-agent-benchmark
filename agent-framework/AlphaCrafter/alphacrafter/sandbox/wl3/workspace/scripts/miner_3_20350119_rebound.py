import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
A=get_account_dict(); syms=A.get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<100:d=get_index_daily_data(s,5000)
 if d is not None and len(d):px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# rebound from 60-day high: assets recently below their high but recovering over 5 days
f=(r.rolling(5).sum() + (P/P.rolling(60).max()-1)).replace([np.inf,-np.inf],np.nan)
ics=[];ns=[];tos=[]
for i in range(65,len(P)-10):
 n=f.columns[f.iloc[i].notna()&P.iloc[i].notna()&P.iloc[i+10].notna()]
 if len(n)<8:continue
 a=f.iloc[i][n];y=P.iloc[i+10][n]/P.iloc[i][n]-1;ics.append((P.index[i],a.corr(y,method='spearman')));ns.append(len(n))
 if i>65:tos.append(np.mean(abs(a.rank(pct=True)-f.iloc[i-1][n].rank(pct=True))))
z=pd.Series(dict(ics)).dropna()
for name,q in [('all',z),('recent120',z.tail(120)),('recent252',z.tail(252)),('recent504',z.tail(504))]: print(name,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('instruments',len(syms),'available',len(P.columns),'avg_valid',round(np.mean(ns),3),'coverage',round(np.mean(ns)/len(syms),4),'turnover',round(np.mean(tos),4),'period',P.index[0],P.index[-1])
for j,q in enumerate(np.array_split(z,4),1): print('block',j,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
pd.DataFrame({'date':[x[0] for x in ics],'factor_ic':[x[1] for x in ics]}).to_csv('scripts/miner_3_20350119_rebound_signal.csv',index=False)

import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
acct=get_account_dict(); syms=acct.get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Downside-conditioned medium momentum: continuation of the 20-day return,
# weighted by the fraction of negative sessions; lagged completed-day signal.
ret20=P.pct_change(20); downfrac=(r<0).rolling(20).mean(); vol=r.rolling(20).std()*np.sqrt(20)
f=(ret20*downfrac/vol).replace([np.inf,-np.inf],np.nan)
ics=[]; ns=[]; tos=[]
for i in range(130,len(P)-10):
 n=f.columns[f.iloc[i].notna()&P.iloc[i].notna()&P.iloc[i+10].notna()]
 if len(n)<8: continue
 a=f.iloc[i][n]; y=P.iloc[i+10][n]/P.iloc[i][n]-1
 ics.append((P.index[i],a.corr(y,method='spearman'))); ns.append(len(n))
 if i>130: tos.append(np.mean(abs(a.rank(pct=True)-f.iloc[i-1][n].rank(pct=True))))
ser=pd.Series(dict(ics)).dropna()
for l,z in [('all',ser),('recent120',ser.tail(120)),('recent252',ser.tail(252)),('recent504',ser.tail(504))]: print(l,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('instruments',len(syms),'available',len(P.columns),'avg_valid',round(np.mean(ns),3),'coverage',round(np.mean(ns)/len(syms),4),'turnover',round(np.mean(tos),4),'period',P.index[0],P.index[-1])
for j,z in enumerate(np.array_split(ser,4),1): print('block',j,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
pd.DataFrame({'date':[x[0] for x in ics],'factor_ic':[x[1] for x in ics]}).to_csv('scripts/miner_2_20350216_downside_momentum20_signal.csv',index=False)

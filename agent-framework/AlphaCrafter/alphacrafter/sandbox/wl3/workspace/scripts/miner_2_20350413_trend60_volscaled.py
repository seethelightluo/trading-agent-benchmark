import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_index_daily_data,get_stock_daily_data
syms=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 if d is not None and len(d): px[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# medium-horizon trend persistence, volatility normalized, with recent shock penalty
vol=r.rolling(60,min_periods=40).std()
f=r.rolling(60,min_periods=40).sum()/vol
# require valid history and evaluate 10d forward returns
ics=[]; ns=[]; tos=[]
for i in range(70,len(P)-10):
 n=f.columns[f.iloc[i].notna()&P.iloc[i].notna()&P.iloc[i+10].notna()]
 if len(n)<8: continue
 a=f.iloc[i][n]; y=P.iloc[i+10][n]/P.iloc[i][n]-1
 c=a.corr(y,method='spearman')
 if pd.notna(c):
  ics.append((P.index[i],c)); ns.append(len(n))
  if i>70:
   old=f.iloc[i-1][n]
   tos.append(np.mean(abs(a.rank(pct=True)-old.rank(pct=True))))
ser=pd.Series(dict(ics)).dropna()
for label,z in [('all',ser),('recent120',ser.tail(120)),('recent252',ser.tail(252)),('recent504',ser.tail(504))]:
 print(label,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('avg_valid',round(np.mean(ns),3),'coverage',round(np.mean(ns)/len(syms),4),'turnover',round(np.mean(tos),4),'period',P.index[0],P.index[-1],'instruments',len(P.columns))
pd.DataFrame({'date':[x[0] for x in ics],'factor_ic':[x[1] for x in ics]}).to_csv('scripts/miner_2_20350413_trend60_volscaled_signal.csv',index=False)
# per broad blocks for regime robustness
for j in range(0,len(ser),300):
 z=ser.iloc[j:j+300]
 if len(z): print('block',j,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))

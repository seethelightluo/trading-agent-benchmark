import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(symbol=s,days=5000)
 if d is not None and len(d)>100:
  d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').drop_duplicates('date')
  frames[s]=d.set_index('date')
# gap reversal, signal known after day t close: use t gap and close return, then forward close return
rows=[]
for s,d in frames.items():
 d['gap']=d['open']/d['close'].shift(1)-1
 d['atr']=((d['high']-d['low'])/d['close'].shift(1)).rolling(20,min_periods=10).mean()
 d['sig']=(-d['gap']/d['atr']).shift(1)
 d['f1']=d['close'].pct_change().shift(-1)
 d['f3']=d['close'].shift(-3)/d['close']-1
 for dt,r in d[['sig','f1','f3']].dropna().iterrows(): rows.append((dt,s,r.sig,r.f1,r.f3))
x=pd.DataFrame(rows,columns=['date','symbol','sig','f1','f3'])
ics=[]
for dt,g in x.groupby('date'):
 if len(g)>=8:
  q=[]
  for h in ['f1','f3']:
   q.append(g['sig'].corr(g[h],method='spearman'))
  ics.append([dt,len(g),*q])
i=pd.DataFrame(ics,columns=['date','n','ic1','ic3']).dropna()
print('dates',len(i),'avg_n',i.n.mean(),'period',i.date.min(),i.date.max())
for c in ['ic1','ic3']:
 z=i[c]; print(c,'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
 for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  zz=z[(i.date>=a)&(i.date<=b+'-12-31')]; print(a,b,len(zz),round(zz.mean(),5),round(zz.mean()/zz.std(ddof=1),4) if len(zz)>1 else None)
print('coverage',len(x)/ (len(i)*15),'turnover_proxy',x.sort_values(['symbol','date']).groupby('symbol').sig.diff().abs().mean())
x.to_csv('scripts/miner_2_20300919_gap_reversal_signal.csv',index=False)

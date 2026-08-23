import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,3000)
 if d is not None:
  q=d[['date','close']].copy();q['symbol']=s;rows.append(q)
px=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index(); r=px.pct_change()
# Novel: directional efficiency over 10 days, strengthened when 30d volatility is below 90d baseline.
net=px/px.shift(10)-1; path=r.abs().rolling(10).sum(); eff=net/path
v30=r.rolling(30).std(); v90=r.rolling(90).std(); f=eff*(v90/v30); sig=f.shift(1)
print('rows',len(px),'assets',px.shape[1])
for h in [1,5,10]:
 fw=px.shift(-h)/px-1;out=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:out.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 D=pd.DataFrame(out,columns=['date','ic','n']).set_index('date').dropna();print('h',h,'dates',len(D),'avg_n',D.n.mean(),'coverage',D.n.sum()/(len(D)*15),'IC',D.ic.mean(),'ICIR',D.ic.mean()/D.ic.std(),'hit',(D.ic>0).mean())
 if h==1:
  for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
   q=D.loc[a:b];print('regime',a,b,len(q),q.ic.mean() if len(q) else np.nan,q.ic.mean()/q.ic.std() if len(q)>1 else np.nan)
  print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean());D.to_csv('scripts/miner_2_20270707_efficiency_state_signal.csv')

import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
    d=get_stock_daily_data(s,4000)
    if d is None or len(d)<30: d=get_index_daily_data(s,4000)
    if d is None: continue
    d=d.sort_values('date').copy(); d['r']=d.close.pct_change()
    # low realized volatility, known only at t-1, predict next daily return
    d['f']=-(d.r.rolling(20,min_periods=15).std()).shift(1)
    d['fr']=d.r.shift(-1)
    d['date']=pd.to_datetime(d.date).dt.strftime('%Y-%m-%d')
    rows.append(d[['date','f','fr']].assign(symbol=s))
x=pd.concat(rows)
ics=[]; cov=[]; turns=[]
for dt,g in x.groupby('date'):
    g=g.replace([np.inf,-np.inf],np.nan).dropna()
    if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
        ics.append(g.f.corr(g.fr)); cov.append(len(g)/15)
        turns.append(g.f.rank(pct=True).sub(0.5).abs().mean())
ics=pd.Series(ics).dropna()
print('dates',len(ics),'avg_n',np.mean([len(g.dropna()) for _,g in x.groupby('date')]),'coverage',np.mean(cov))
print('IC %.8f ICIR %.6f hit %.4f'% (ics.mean(),ics.mean()/ics.std(),(ics>0).mean()))
print('recent IC',ics.tail(500).mean(),'first/last',ics.head(500).mean(),ics.tail(500).mean())
# horizon decay
for h in [1,3,5,10]:
 z=[]
 for s,g in x.groupby('symbol'):
  # reconstruct forward from raw unavailable; use fetch again
  d=get_stock_daily_data(s,4000)
  if d is None: d=get_index_daily_data(s,4000)
  d=d.sort_values('date'); r=d.close.pct_change(); f=(-(r.rolling(20,min_periods=15).std())).shift(1); fr=d.close.shift(-h)/d.close-1
  q=pd.DataFrame({'date':pd.to_datetime(d.date).dt.strftime('%Y-%m-%d'),'f':f,'fr':fr,'symbol':s}); z.append(q)
 q=pd.concat(z); a=[]
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8 and g.f.nunique()>1:a.append(g.f.corr(g.fr))
 print('h',h,'IC',pd.Series(a).dropna().mean(),'n',len(a))
# signal artifact
out=x[['date','symbol','f']].dropna(); out.to_csv('scripts/miner_1_20271118_lowvol_signal.csv',index=False)

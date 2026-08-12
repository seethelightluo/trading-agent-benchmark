import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,days=1800)
 if d is None or len(d)<120:d=get_index_daily_data(s,days=1800)
 if d is not None:
  x=d[['date','close']].copy();x['symbol']=s;rows.append(x)
p=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill();r=np.log(p).diff()
# volatility carry: low realized volatility plus positive medium trend, lagged
v20=r.rolling(20).std(); v60=r.rolling(60).std(); tr=p.pct_change(40)
f=((tr.rank(axis=1,pct=True)) - (v20/v60).rank(axis=1,pct=True)).shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; a=[];n=[];t=[];prev=None
 for i in range(len(p)-h):
  ok=f.iloc[i].notna()&y.iloc[i].notna()
  if ok.sum()>=8:
   a.append(f.iloc[i][ok].corr(y.iloc[i][ok]));n.append(ok.sum());q=f.iloc[i].rank(pct=True);t.append(np.nan if prev is None else (q-prev).abs().mean());prev=q
 a=np.array([z for z in a if np.isfinite(z)]);print('h',h,'dates',len(a),'avg_n',round(np.mean(n),3),'coverage',round(np.mean(n)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(t),4))
 if h==10:
  for lo,hi in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-03-18')]:
   z=[]
   for dt in p.loc[lo:hi].index:
    ok=f.loc[dt].notna()&y.loc[dt].notna()
    if ok.sum()>=8:z.append(f.loc[dt][ok].corr(y.loc[dt][ok]))
   z=np.array([q for q in z if np.isfinite(q)]);print('regime',lo[:4],len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20320318_volcarry_signal.csv',index=False)

import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
U=[s for s in U if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100: d=get_index_daily_data(s,4000)
 if d is not None: D[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
P=pd.DataFrame(D).sort_index(); r=P.pct_change();
# Recovery efficiency: distance above trailing 60d low, normalized by trailing 20d volatility.
# Lag one completed bar; higher values mean recovery from drawdown achieved with less risk.
low=P.rolling(60,min_periods=40).min(); vol=r.rolling(20,min_periods=15).std(); f=((P/low-1)/(vol*np.sqrt(20))).shift(1)
rows=[]; sig=[]
for h in [1,3,5,10,20]:
 vals=[]; ns=[]; dates=[]
 fr=P.pct_change(h).shift(-h)
 for dt in P.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 q=pd.Series(vals,index=dates).dropna(); print('H',h,'dates',len(q),'avgN',round(float(np.mean(ns)),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),3))
 if h==5:
  for dt in q.index:
   for s in f.columns:
    sig.append({'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':f.loc[dt,s]})
print('coverage',round(f.notna().mean().mean(),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6),'instruments',len(P.columns),'span',P.index.min(),P.index.max())
pd.DataFrame(sig).to_csv('scripts/miner_2_20280907_recovery_efficiency_signal.csv',index=False)

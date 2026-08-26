import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# VIX is observation-only and used solely as a regime input
v=get_index_daily_data('VIX',days=2500)
v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].astype(float)
prices={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  d['date']=pd.to_datetime(d['date']); prices[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(prices).sort_index(); r=p.pct_change()
# risk-on condition known at t: VIX below its trailing 60-session median; signal is 20d momentum / 40d vol
vr=v.reindex(p.index).ffill(); gate=(vr < vr.rolling(60,min_periods=60).median()).astype(float)
ret=p/p.shift(20)-1; vol=r.rolling(40,min_periods=40).std()*np.sqrt(20)
sig=-ret/vol.replace(0,np.nan)*gate.values[:,None]
# cross-sectional demean, only dates with >=8 valid names
sig=sig.sub(sig.mean(axis=1),axis=0)
rows=[]
for h in [5,10,20,40]:
 fwd=p.shift(-h)/p-1
 for dt in sig.index:
  x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(ic): rows.append((dt,h,len(z),ic))
 out=pd.DataFrame([x for x in rows if x[1]==h],columns=['date','h','n','ic'])
 ic=out.ic.mean(); sd=out.ic.std(ddof=1); icir=ic/sd*np.sqrt(len(out)) if sd else np.nan
 print(f'h={h} dates={len(out)} avg_n={out.n.mean():.2f} coverage={out.n.sum()/(len(out)*len(U)):.4f} IC={ic:.6f} ICIR={icir:.6f} hit={((out.ic>0).mean()):.4f}')
# artifact at all dates for audit
sig.to_csv('scripts/miner_1_20281120_vix_gated_momentum_signal.csv',index_label='date')
print('assets',len(prices),'rows',len(p),'gate_on',gate.mean())

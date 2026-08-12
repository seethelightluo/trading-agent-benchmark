import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
cut=pd.Timestamp('2029-11-29'); watch=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in watch:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d)>120:
  d=d.copy(); d.date=pd.to_datetime(d.date); frames[s]=d[d.date<=cut].set_index('date').sort_index()
close=pd.concat({s:d.close for s,d in frames.items()},axis=1).sort_index(); high=pd.concat({s:d.high for s,d in frames.items()},axis=1).reindex(close.index); low=pd.concat({s:d.low for s,d in frames.items()},axis=1).reindex(close.index)
rng=(high-low).replace(0,np.nan); clv=((close-low)/rng-0.5).clip(-1,1); raw=clv.rolling(3,min_periods=3).mean(); raw=raw.sub(raw.mean(axis=1),axis=0).div(raw.std(axis=1).replace(0,np.nan),axis=0); f=-raw
out=[]
for h in [1,3,5,10]:
 fr=close.shift(-h)/close-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.asarray(vals,float); a=a[np.isfinite(a)]; out.append((h,len(a),float(a.mean()),float(a.mean()/(a.std(ddof=1)+1e-12)),float((a>0).mean())))
print('assets',len(frames),'dates',len(close),'valid_dates_h1',out[0][1],'avg_n',np.mean([f.loc[d].notna().sum() for d in f.index]))
for x in out: print('H',x[0],'N',x[1],'IC %.8f ICIR %.8f hit %.4f'%x[2:])
r=f.rank(axis=1,pct=True); print('turnover_proxy %.6f coverage %.6f'%((r.diff().abs().mean(axis=1)).mean(),f.notna().sum().sum()/f.size))
for start in ['2028-01-01','2029-01-01','2029-07-01']:
 z=[]
 for dt in f.index:
  if dt<pd.Timestamp(start): continue
  q=pd.concat([f.loc[dt],(close.shift(-1)/close-1).loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 z=np.asarray(z); z=z[np.isfinite(z)]; print(start,'N',len(z),'IC',float(z.mean()),'ICIR',float(z.mean()/(z.std(ddof=1)+1e-12)))
f.to_csv('scripts/miner_2_20291129_candle_location_reversal_1d_signal.csv')

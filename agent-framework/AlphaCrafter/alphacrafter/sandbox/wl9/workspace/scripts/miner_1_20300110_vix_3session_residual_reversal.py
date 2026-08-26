import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)==0:d=get_index_daily_data(s,3000)
 if d is not None and len(d):px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().ffill(); ret=P.pct_change()
v=get_index_daily_data('VIX',3000)
if v is None:v=get_stock_daily_data('VIX',3000)
V=v.set_index('date').close.astype(float).reindex(P.index).ffill()
rows=[]
for i in range(80,len(P)-21):
 hist=V.iloc[i-61:i]; med=hist.median()
 if len(hist)<60 or (V.iloc[i-1]<=med or V.iloc[i-2]<=med or V.iloc[i-3]<=med):continue
 r10=ret.iloc[i-10:i].sum(); resid=r10-r10.median(); down=ret.iloc[i-20:i].clip(upper=0).std()
 sig=-(resid/down.replace(0,np.nan)); z=sig.rank(pct=True); f=ret.iloc[i:i+5].sum(); q=z.dropna().index.intersection(f.dropna().index)
 rows.append({'date':P.index[i],'ic':z[q].corr(f[q]) if len(q)>=8 else np.nan,'n':z.notna().sum(),**{'s_'+s:sig.get(s,np.nan) for s in U}})
R=pd.DataFrame(rows).set_index('date'); a=R.ic.dropna()
print('dates',len(R),'ic_dates',len(a),'avg_n',R.n.mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',R.n.mean()/15)
print('recent_2027_30',R.loc['2027':].ic.mean(),R.loc['2027':].ic.count(), 'recent_icir',R.loc['2027':].ic.mean()/R.loc['2027':].ic.std(ddof=1))
rank=R[['s_'+s for s in U]].rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean().mean())
R.to_csv('scripts/miner_1_20300110_vix_3session_residual_reversal_5d_signal.csv')

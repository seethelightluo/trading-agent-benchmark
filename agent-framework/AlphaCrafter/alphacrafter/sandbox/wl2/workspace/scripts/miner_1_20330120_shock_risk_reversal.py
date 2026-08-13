import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=5000)
 if d is None or len(d)<100:return None
 d=d.copy();d['date']=pd.to_datetime(d['date'])
 return d.drop_duplicates('date').set_index('date')['close'].astype(float)
px={s:load(s) for s in U};px={s:x for s,x in px.items() if x is not None};P=pd.DataFrame(px).sort_index();R=P.pct_change();
# Contrarian shock: reverse recent 5-session return, scaled by trailing 20-session risk; all inputs lagged.
f=(-P.pct_change(5)/R.rolling(20).std()).shift(1)
rows={h:[] for h in [1,3,5,10,20]}; dates=[]; cov=[]; turn=[]
for i in range(len(P)-20):
 x=f.iloc[i]
 if x.notna().sum()>=8:
  dates.append(P.index[i]);cov.append(x.notna().mean())
  if i: turn.append((x.rank(pct=True)-f.iloc[i-1].rank(pct=True)).abs().mean())
 for h in rows:
  z=pd.concat([x,P.iloc[i+h].div(P.iloc[i])-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: rows[h].append(z.iloc[:,0].corr(z.iloc[:,1]))
print('dates',len(dates),'assets',len(P.columns),'coverage',round(np.mean(cov),4),'turnover',round(np.mean(turn),4))
for h,v in rows.items():
 a=np.asarray(v);print('h',h,'n',len(a),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
for label,mask in [('early',np.array(dates)<pd.Timestamp('2026-07-16')),('online',np.array(dates)>=pd.Timestamp('2026-07-16'))]:
 v=[]
 for dt in np.array(dates)[mask]:
  i=P.index.get_loc(dt);z=pd.concat([f.loc[dt],P.iloc[i+10].div(P.iloc[i])-1],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print(label,'dates',len(v),'IC',round(np.nanmean(v),6),'ICIR',round(np.nanmean(v)/np.nanstd(v,ddof=1),6))
f.index.name='date';f.to_csv('scripts/miner_1_20330120_shock_risk_reversal_signal.csv')

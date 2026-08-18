import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    try: d=get_stock_daily_data(s, days=6000)
    except Exception: d=None
    if d is not None and len(d)>200:
        frames[s]=d.sort_values('date').set_index('date')['close'].astype(float)
px=pd.concat(frames,axis=1).sort_index(); r=np.log(px).diff()
rv20=r.rolling(20,min_periods=15).std(); rv60=r.rolling(60,min_periods=45).std()
# Candidate: volatility-compression-confirmed medium momentum, lagged causally
f=(px.pct_change(40)*(1-(rv20/rv60)).clip(-1,1)).shift(1)
rows=[]
for i in range(len(px)-10):
 z=pd.concat([f.iloc[i],px.iloc[i+10]/px.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((px.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
ser=pd.Series([x[1] for x in rows]); print('candidate compression_trend'); print('dates',len(ser),'avgN',np.mean([x[2] for x in rows]),'coverage',f.notna().sum().sum()/f.size,'IC10',ser.mean(),'daily_ICIR',ser.mean()/ser.std(ddof=1),'hit',np.mean(ser>0),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [5,10,20,40]:
 vals=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.mean(vals),len(vals))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=[v for d,v,n in rows if a<=str(d.year)<=b]; print('regime',a,b,len(q),np.mean(q) if q else None)
out=f.dropna(how='all').tail(600).reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20351026_compression_trend_signal.csv',index=False)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20351026_compression_trend_ic.csv',index=False)
print('artifacts written')

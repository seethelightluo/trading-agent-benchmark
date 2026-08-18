import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 try:
  d=get_stock_daily_data(s,days=6000)
  if d is not None and len(d)>250: fs[s]=d.sort_values('date').set_index('date')['close'].astype(float)
 except: pass
px=pd.concat(fs,axis=1).sort_index(); r=np.log(px).diff(); vol=r.rolling(20,min_periods=15).std();
# short-term reversal scaled by idiosyncratic volatility, with medium trend damping
rev=(-px.pct_change(5)/(vol*np.sqrt(5))).clip(-5,5); trend=px.pct_change(60); f=(rev*(1-0.35*trend.rank(axis=1,pct=True))).shift(1)
rows=[]
for i in range(len(px)-10):
 z=pd.concat([f.iloc[i],px.iloc[i+10]/px.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): rows.append((px.index[i],c,len(z)))
ser=pd.Series([x[1] for x in rows]); print('loaded',len(fs),'dates',len(ser),'avgN',np.mean([x[2] for x in rows]),'IC10',ser.mean(),'ICIR',ser.mean()/ser.std(ddof=1),'hit',np.mean(ser>0),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [5,10,20,40]:
 vals=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c)
 print('decay',h,np.mean(vals),len(vals))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=[v for d,v,n in rows if a<=str(d.year)<=b]; print('regime',a,b,len(q),np.mean(q) if q else None)
f.dropna(how='all').tail(500).reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350914_reversal_signal.csv',index=False)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20350914_reversal_ic.csv',index=False)

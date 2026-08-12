import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
U=[x for x in U if x not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100: d=get_index_daily_data(s,4000)
 if d is not None:
  d=d.copy(); d['date']=pd.to_datetime(d['date']); D[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(D).sort_index(); r=P.pct_change()
# volatility-compression breakout: medium trend scaled by recent risk and rewarded when risk is compressing
rv20=r.rolling(20).std(); rv60=r.rolling(60).std()
f=(P.pct_change(20)/(rv20*np.sqrt(20))).mul((rv60/rv20).clip(0.5,2.0))
# lag completed bar
f=f.shift(1)
print('universe',len(U),'dates',len(P),'range',P.index.min(),P.index.max())
for h in [1,3,5,10,20]:
 ic=[]; cov=[]; turnovers=[]
 for i in range(len(P)-h):
  a=f.iloc[i]; y=P.pct_change(h).iloc[i+h]
  z=pd.concat([a,y],axis=1).dropna()
  if len(z)>=8:
   ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); cov.append(len(z)/len(U))
  if i>0:
   q=f.iloc[i].rank(pct=True); q0=f.iloc[i-1].rank(pct=True)
   turnovers.append((q-q0).abs().mean())
 x=pd.Series(ic).dropna(); print('h',h,'dates',len(x),'avgN',round(np.mean([c*len(U) for c in cov]),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),3),'turn',round(np.nanmean(turnovers),4),'coverage',round(np.mean(cov),3))
# regime split 5d
h=5; ic=[]
for i in range(len(P)-h):
 z=pd.concat([f.iloc[i],P.pct_change(h).iloc[i+h]],axis=1).dropna()
 if len(z)>=8: ic.append((P.index[i],z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
x=pd.DataFrame(ic,columns=['date','ic']).set_index('date');
for name,mask in [('2020-22',x.index<'2023'),('2023-25',(x.index>='2023')&(x.index<'2026')),('2026-28',x.index>='2026')]:
 q=x.loc[mask,'ic']; print(name,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None)

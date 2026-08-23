import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-10-08'); fs={}
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)==0:d=get_index_daily_data(s,3000)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);d=d[d.date<=cut].sort_values('date').set_index('date');d['r']=d.close.pct_change();fs[s]=d
rows=[]
for s,d in fs.items():
 # volume activity confirms medium-term trend; all inputs completed at t
 vol=d.volume.replace(0,np.nan); activity=vol.rolling(5).mean()/vol.rolling(40).mean()
 sig=d.close.pct_change(20)/d.r.rolling(20).std()*np.log(activity.clip(0.5,2.0))
 # blend keeps signal directional but volume confirmation is interpretable
 sig=(d.close.pct_change(20)/ (d.r.rolling(60).std()*np.sqrt(20)))*(0.5+0.5*activity.rank(pct=True))
 fwd=d.close.pct_change().shift(-1)
 for dt in d.index:
  if pd.notna(sig.get(dt)) and pd.notna(fwd.get(dt)):rows.append((dt,s,sig.loc[dt],fwd.loc[dt]))
x=pd.DataFrame(rows,columns=['date','symbol','f','r']);ics=x.groupby('date').apply(lambda z:z.f.corr(z.r),include_groups=False).dropna()
print('volume_confirmed_risk_momentum; dates',len(ics),'avg_names',x.groupby('date').size().mean(),'coverage',x.groupby('date').size().mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/ics.std(),(ics>0).mean()))
for h in [5,10,20]:
 a=[]
 for s,d in fs.items():
  act=d.volume.rolling(5).mean()/d.volume.rolling(40).mean(); sig=d.close.pct_change(20)/(d.r.rolling(60).std()*np.sqrt(20))*(.5+.5*act.rank(pct=True)); rr=d.close.pct_change(h).shift(-h);z=pd.DataFrame({'f':sig,'r':rr}).dropna()
  a += [(dt,z.loc[dt,'f'],z.loc[dt,'r']) for dt in z.index]
 a=pd.DataFrame(a,columns=['date','f','r']).groupby('date').apply(lambda z:z.f.corr(z.r),include_groups=False).dropna();print('horizon',h,'ICIR',a.mean()/a.std(),'IC',a.mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 a=ics.loc[lo:hi];print('regime',lo,hi,len(a),a.mean(),a.mean()/a.std())

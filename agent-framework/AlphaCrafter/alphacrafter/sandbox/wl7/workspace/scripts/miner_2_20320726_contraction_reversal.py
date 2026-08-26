import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; frames={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,days=2600)
  except Exception:pass
  if d is not None:break
 if d is not None and len(d)>100:
  d=d.copy();d['date']=pd.to_datetime(d['date']);frames[s]=d.sort_values('date').set_index('date')
close=pd.DataFrame({s:d['close'] for s,d in frames.items()}).sort_index();ret=np.log(close).diff();r5=np.log(close/close.shift(5));v20=ret.rolling(20,min_periods=15).std()*np.sqrt(252);v60=ret.rolling(60,min_periods=40).std()*np.sqrt(252)
f=(-(r5/v20)*(v20/v60).clip(.5,1.5)).sub((-(r5/v20)*(v20/v60).clip(.5,1.5)).median(axis=1),axis=0); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],np.log(close.shift(-10)/close).loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),len(z)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('universe',len(frames),'dates',len(o),'avgN',round(o.n.mean(),2),'coverage',round(f.stack().notna().mean(),4));print('H10 IC',round(o.ic.mean(),6),'ICIR',round(o.ic.mean()/o.ic.std(ddof=1),6),'hit',round((o.ic>0).mean(),4));print('thirds',[round(o.iloc[a:b].ic.mean(),6) for a,b in [(0,len(o)//3),(len(o)//3,2*len(o)//3),(2*len(o)//3,len(o))]]);print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [1,5,20]:
 rr=[];fw=np.log(close.shift(-h)/close)
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:rr.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('H',h,'n',len(rr),'IC',round(np.nanmean(rr),6),'ICIR',round(np.nanmean(rr)/np.nanstd(rr,ddof=1),6))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20320726_contraction_reversal_signal.csv',index=False);print('artifact written')

import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];U=[s for s in U if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}];D={}
for s in U:
 d=get_stock_daily_data(s,2000)
 if d is None or len(d)<120:d=get_index_daily_data(s,2000)
 if d is not None and len(d)>=120:D[s]=d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill();r=np.log(px).diff();r5=r.rolling(5).sum();med=r5.median(axis=1);disp=r5.sub(med,axis=0).abs().median(axis=1);thr=disp.rolling(60,min_periods=30).median();f=(-(r5.sub(med,axis=0))*(disp/thr).clip(upper=3)).shift(1)
print('range',px.index.min().date(),px.index.max().date(),'assets',len(D),'dates',len(px),flush=True)
for h in [1,5,10,20]:
 fr=px.shift(-h)/px-1;a=[];ns=[];turn=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()));ns.append(len(z))
  q=pd.concat([f.loc[dt],f.shift(1).loc[dt]],axis=1).dropna()
  if len(q)>=8:turn.append(1-q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 a=np.array(a);print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(turn),4),flush=True)
f.index.name='date';f.to_csv('scripts/miner_2_20290208_dispersion_reversal_signal.csv')

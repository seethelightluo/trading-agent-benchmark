import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=3000)
 if d is not None and len(d)>150:
  d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.sort_index()
px=pd.concat(P,axis=1).sort_index().ffill();r=px.pct_change()
# Cross-asset stress-conditioned reversal: fade recent losses only when macro stress is elevated;
# in calm regimes use the slower 20d trend, yielding an interpretable state switch.
def load(name):
 import os
 for p in ['../persistent/index_data/'+name+'.csv','../persistent/index_data/'+name+'.CSV']:
  if os.path.exists(p):
   z=pd.read_csv(p);z['date']=pd.to_datetime(z['date']);return z.set_index('date')['close'].reindex(px.index).ffill()
v=load('VIX');dxy=load('DXY')
stress=(v>v.rolling(60,min_periods=30).median()) & (v.pct_change(5)>0)
rev=-r.rolling(5,min_periods=5).sum(); trend=np.log1p(r.clip(lower=-.99)).rolling(20,min_periods=15).sum()
# macro stress also recognizes rising dollar; all inputs lagged through shift
f=rev.where(stress,trend).div(r.rolling(20,min_periods=15).std()*np.sqrt(20)).shift(1)
print('universe',len(U),'loaded',len(P),'dates',len(px),'stress',round(stress.mean(),4))
for h in [1,5,10]:
 y=px.pct_change(h).shift(-h); vals=[];ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q):vals.append(q);ns.append(len(a))
 x=pd.Series(vals);print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20280727_regime_switch_signal.csv',index=False)

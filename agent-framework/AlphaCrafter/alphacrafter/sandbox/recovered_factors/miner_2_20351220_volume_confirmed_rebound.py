"""miner_2 20351220: validate volume-confirmed medium-term rebound."""
import json, numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']; ps={};vs={}
for a in assets:
 try:
  d=get_stock_daily_data(a,6000)
  if 'date' in d.columns:d=d.set_index('date')
  d.index=pd.to_datetime(d.index)
  ps[a]=pd.to_numeric(d['close'],errors='coerce'); vs[a]=pd.to_numeric(d['volume'],errors='coerce') if 'volume' in d else pd.Series(index=d.index,dtype=float)
 except Exception as e: print('ERR',a,e)
px=pd.DataFrame(ps).sort_index(); vol=pd.DataFrame(vs).reindex(px.index)
r=px.pct_change(fill_method=None)
# Rebound score: negative 20d return, strengthened when recent participation exceeds its slow baseline.
particip=(vol/vol.rolling(60,min_periods=30).median()).clip(0.25,4)
factor=-(px.pct_change(20,fill_method=None))*particip
# only information through t-1; score at t predicts t->t+h
factor=factor.shift(1)
def spe(a,b):
 z=pd.concat([a,b],axis=1).dropna()
 return np.nan if len(z)<8 else z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
def calc(h):
 fw=px.shift(-h)/px-1; ic=pd.Series([spe(factor.iloc[i],fw.iloc[i]) for i in range(len(px))],index=px.index).dropna()
 tos=[]
 for i in range(1,len(factor)):
  z=pd.concat([factor.iloc[i-1],factor.iloc[i]],axis=1).dropna()
  if len(z)>=8:tos.append(1-spe(z.iloc[:,0],z.iloc[:,1]))
 sd=ic.std(ddof=1);return dict(horizon_days=h,daily_paper_ic=float(ic.mean()),daily_paper_icir=float(ic.mean()/sd),ic_std=float(sd),ic_hit_ratio=float((ic>0).mean()),n_dates=len(ic),turnover=float(np.mean(tos)),coverage=float(factor.notna().sum().sum()/factor.size),mean_valid_instruments=float(factor.notna().sum(axis=1).mean()),yearly_ic={str(y):float(x.mean()) for y,x in ic.groupby(ic.index.year)})
print('PANEL',px.shape,'dates',px.index.min(),px.index.max(),'assets',len(px.columns))
for h in (1,5,10,20):print('METRIC',json.dumps(calc(h),sort_keys=True))
factor.to_pickle('scripts/miner_2_20351220_volume_confirmed_rebound_signal.pkl')

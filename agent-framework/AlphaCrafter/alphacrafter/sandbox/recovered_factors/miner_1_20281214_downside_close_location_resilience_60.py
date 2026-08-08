"""One factor idea: conditional downside close-location resilience, 60d.
On own negative-return days, measure where close finished in day's range; high score
means repeated recovery away from intraday lows under stress."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; O={};H={};L={};C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 for dest,col in [(O,'open'),(H,'high'),(L,'low'),(C,'close')]: dest[a]=pd.to_numeric(d[col],errors='coerce')
o,h,l,c=map(pd.DataFrame,(O,H,L,C)); r=c.pct_change(); rng=(h-l).replace(0,np.nan)
loc=((c-l)/rng).clip(0,1)
# Lag rolling statistic so each score uses completed observations only.
f=loc.where(r<0).rolling(60,min_periods=12).mean().shift(1)
f=f.sub(f.median(axis=1),axis=0); cutoff=c.dropna(how='all').index.max(); FW={x:c.shift(-x)/c-1 for x in [1,5,10,20]}
def ev(k,span=None):
 x=f if span is None else f.loc[span[0]:span[1]];y=FW[k].reindex(x.index);z=[];nn=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);nn.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(nn)),2),'min_n':int(min(nn))}
print('FACTOR downside_close_location_resilience_60 cutoff',cutoff.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
for k in [1,5,10,20]:print('H',k,ev(k))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))

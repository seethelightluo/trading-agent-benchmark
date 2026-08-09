"""One idea: downside-versus-upside close-location asymmetry, trailing 60 observations."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; O={};Hh={};Ll={};C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 O[a]=pd.to_numeric(d.open,errors='coerce');Hh[a]=pd.to_numeric(d.high,errors='coerce');Ll[a]=pd.to_numeric(d.low,errors='coerce');C[a]=pd.to_numeric(d.close,errors='coerce')
o=pd.DataFrame(O); hi=pd.DataFrame(Hh);lo=pd.DataFrame(Ll);p=pd.DataFrame(C)
r=p.pct_change(); loc=(p-lo)/(hi-lo).replace(0,np.nan)
# Positive scores: closes relatively stronger on adverse days than on advancing days.
down=loc.where(r<0).rolling(60,min_periods=12).mean(); up=loc.where(r>0).rolling(60,min_periods=12).mean()
f=down-up; f=f.sub(f.median(axis=1),axis=0)
cut=p.dropna(how='all').index.max(); hs=[1,5,10,20]; fw={h:p.shift(-h)/p-1 for h in hs}
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]];y=fw[h].reindex(x.index);z=[];nn=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);nn.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(nn)),2),'min_n':int(min(nn))}
print('FACTOR downside_upside_close_location_asymmetry_60 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
for h in hs:print('H',h,ev(h))
for n,s in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01',str(cut.date()))),('recent180',(str(cut-pd.Timedelta(days=180)),str(cut.date())))]:print('REGIME10',n,ev(10,s))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))

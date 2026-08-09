import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
# completed daily panels only
px={}; vol={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
 px[a]=pd.to_numeric(d['close'],errors='coerce'); vol[a]=pd.to_numeric(d.get('volume',np.nan),errors='coerce')
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vol).reindex(P.index)
R=P.pct_change(); peer=R.sub(R.median(axis=1),axis=0)
# idiosyncratic downside event: return below each asset's trailing 60d 25th percentile; all inputs shifted one full day
q=peer.rolling(60,min_periods=45).quantile(.25).shift(1)
event=(peer<q).astype(float).shift(1)
# Require a calm/common market: cross-section dispersion at date was below its 60d median.
# In calm common tapes recurrent idiosyncratic downside is a deterioration signal.
disp=R.std(axis=1); calm=(disp<disp.rolling(60,min_periods=45).median()).shift(1).astype(float)
# recent 20d clustering; inverse orientation rewards assets without repeated idiosyncratic downside
F=-(event.mul(calm,axis=0).rolling(20,min_periods=15).mean())
# exclude mechanically all-zero dates only through cross-sectional rank correlation
print('IDEA inverse calm-regime idiosyncratic downside-event clustering (60q/20cluster)')
print('dates',P.index.min().date(),P.index.max().date(),'assets',len(A),'cells',int(F.notna().sum().sum()),'coverage',round(F.notna().mean().mean(),4))
ics={}
for h in [1,5,10,20]:
 fr=P.shift(-h)/P-1; rows=[]; breadth=[]
 for t in F.index:
  x=F.loc[t]; y=fr.loc[t]; z=pd.concat([x,y],axis=1).dropna()
  # adequate variability plus required breadth
  if len(z)>=8 and z.iloc[:,0].nunique()>=3 and z.iloc[:,1].nunique()>=3:
   rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); breadth.append(len(z))
 s=pd.Series(rows).dropna(); ics[h]=s
 print('h',h,'ic_dates',len(s),'mean_ic',round(s.mean(),6),'icir',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'mean_n',round(np.mean(breadth),2))
# turnover ranks (consecutive eligible panel days)
ranks=F.rank(axis=1,pct=True); print('turnover',round(ranks.diff().abs().mean(axis=1).mean(),6))
# 10d regimes
sigs=ics[10]
for label, mask in [('2025_26',(sigs.index>='2025-01-01')&(sigs.index<'2027-01-01')),('2027_current',sigs.index>='2027-01-01'),('recent180',sigs.index>=P.index.max()-pd.Timedelta(days=180))]:
 z=sigs[mask];print(label,'n',len(z),'ic',round(z.mean(),6),'icir',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('cutoff',P.index.max().date())

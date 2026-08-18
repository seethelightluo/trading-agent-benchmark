import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data,get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,6000)
 if d is None or len(d)<100: d=get_index_daily_data(s,6000)
 if d is not None and len(d):
  x=d[['date','close']].dropna(); x.date=pd.to_datetime(x.date); px[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
P=pd.concat(px,axis=1).sort_index(); r=P.pct_change(); v20=r.rolling(20,min_periods=12).std()*np.sqrt(252)
# Candidate: short trend curvature, normalized by medium risk; all inputs lagged at decision date.
f=P.pct_change(5)/v20 - P.pct_change(20)/v20
fr=P.shift(-10)/P-1
obs=[]; counts=[]; turns=[]
for i,dt in enumerate(f.index[:-10]):
 a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  obs.append(a[ok].corr(b[ok],method='spearman')); counts.append(int(ok.sum()))
  if i:
   prev=f.iloc[i-1]; kk=ok&prev.notna()
   if kk.sum()>=8: turns.append((a[kk].rank(pct=True)-prev[kk].rank(pct=True)).abs().mean())
q=pd.Series(obs).dropna(); print('ALL',{'dates':len(q),'avg_names':round(np.mean(counts),3),'coverage':round(np.mean(counts)/15,4),'IC':round(q.mean(),6),'ICIR':round(q.mean()/q.std(ddof=1),6),'hit':round((q>0).mean(),4),'turnover':round(np.mean(turns),4),'start':str(f.index.min().date()),'end':str(f.index.max().date())})
for k in [120,252,504]:
 z=q.tail(k); print('RECENT',k,{'dates':len(z),'IC':round(z.mean(),6),'ICIR':round(z.mean()/z.std(ddof=1),6)})
for h in [1,5,10,20]:
 vals=[]
 for dt in f.index[:-h]:
  a=f.loc[dt]; b=(P.shift(-h)/P-1).loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(a[ok].corr(b[ok],method='spearman'))
 print('DECAY',h,len(vals),round(float(np.nanmean(vals)),6))
pd.DataFrame({'date':f.index[:-10][:len(obs)],'factor_ic':obs}).to_csv('scripts/miner_2_20350302_curvature5_20_signal.csv',index=False)

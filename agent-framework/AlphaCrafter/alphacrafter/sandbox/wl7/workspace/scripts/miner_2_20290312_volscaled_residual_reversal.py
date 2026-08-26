import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:d=fn(s,1200)
  except Exception:d=None
  if d is not None and len(d):break
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);D[s]=x.drop_duplicates('date').set_index('date').close
P=pd.DataFrame(D).sort_index(); r=P.pct_change(); r20=P.pct_change(20); vol=r.rolling(20).std(); med=r20.median(axis=1)
sig=-(r20.sub(med,axis=0)).div(vol.replace(0,np.nan))
rows=[]
for i,d in enumerate(P.index):
 for s in P.columns:
  if pd.notna(sig.loc[d,s]):
   x={'date':d.strftime('%Y-%m-%d'),'symbol':s,'factor':float(sig.loc[d,s])}
   for h in (5,10,20): x[f'fwd{h}']=float(P.iloc[i+h][s]/P.iloc[i][s]-1) if i+h<len(P) and pd.notna(P.iloc[i+h][s]) else np.nan
   rows.append(x)
out=pd.DataFrame(rows);out.to_csv('scripts/miner_2_20290312_volscaled_residual_reversal_signal.csv',index=False)
def evaluate(h):
 vals=[]; cov=[]; turnover=[]
 for i in range(len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));cov.append(len(z)/15)
   if i:
    q=pd.concat([sig.iloc[i-1],sig.iloc[i]],axis=1).dropna()
    turnover.append(1-q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 a=np.array(vals);return len(a),float(np.nanmean(a)),float(np.nanmean(a)/np.nanstd(a,ddof=1)),float(np.mean(a>0)),float(np.mean(cov)),float(np.nanmean(turnover))
for h in (5,10,20): print('H',h,'N IC ICIR hit coverage turnover',evaluate(h))
for lo,hi in [('2025','2026'),('2027','2028'),('2028','2029')]:
 vals=[]
 for i,d in enumerate(P.index):
  if pd.Timestamp(lo+'-01-01')<=d<=pd.Timestamp(hi+'-12-31') and i+10<len(P):
   z=pd.concat([sig.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
   if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=np.array(vals);print('REGIME',lo,hi,len(a),float(np.nanmean(a)),float(np.nanmean(a)/np.nanstd(a,ddof=1)) if len(a)>1 else np.nan)
print('dates',P.index.min(),P.index.max(),'instruments',len(P.columns))

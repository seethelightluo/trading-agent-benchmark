import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<200: d=get_index_daily_data(s,1800)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# Trend persistence quality: signed mean return times directional agreement, divided by realized vol.
# This rewards steady trends and penalizes choppy paths; signal is lagged one completed day.
for w in [20,40,60]:
 sig=(R.rolling(w).mean() * (R.gt(0).rolling(w).mean()-0.5).abs()*2 / (R.rolling(w).std()+1e-8)).shift(1)
 # preserve direction while weighting agreement (not absolute agreement)
 sig=(R.rolling(w).mean() * (R.gt(0).rolling(w).mean()*2-1) / (R.rolling(w).std()+1e-8)).shift(1)
 for h in [1,3,5]:
  fwd=P.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if np.isfinite(c): vals.append(c); ns.append(len(z)); dates.append(dt)
  a=np.array(vals); print('VAR',w,h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),np.mean(a>0),np.mean(ns)/len(U)))
  for lab,cut in [('2027+',pd.Timestamp('2027-01-01')),('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
   q=np.array([v for v,d in zip(vals,dates) if d>=cut]); print(lab,len(q),'%.6f %.6f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12)) if len(q)>1 else 'NA')
 # turnover proxy
 print('turnover',sig.diff().abs().mean().mean())

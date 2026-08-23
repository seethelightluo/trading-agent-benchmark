import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END='2026-11-04'; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 x=x.loc[x.index<=END]
 r=x.close.pct_change(); med=x.volume.rolling(20,min_periods=10).median()
 # volume-surprise reversal: fade unusually large one-day moves, scaled smoothly by log abnormal volume
 sig=-r*np.log1p((x.volume/(med+1e-12)).clip(0.1,20))
 D[s]=pd.DataFrame({'sig':sig,'r1':r.shift(-1),'r5':x.close.pct_change(5).shift(-5)},index=x.index)
all_dates=sorted(set().union(*[set(z.index) for z in D.values()]))
S=pd.DataFrame({s:D[s].sig for s in U}).sort_index(); ranks=S.rank(axis=1,pct=True)
print('period',all_dates[0],all_dates[-1], 'coverage',S.notna().mean().mean(),'turnover',ranks.diff().abs().mean().mean())
for h in ['r1','r5']:
 vals=[]; dates=[]; ns=[]
 for dt in all_dates:
  z=pd.DataFrame({'x':[D[s].at[dt,'sig'] if dt in D[s].index else np.nan for s in U], 'y':[D[s].at[dt,h] if dt in D[s].index else np.nan for s in U]},index=U).dropna()
  if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.x,z.y).statistic);dates.append(dt);ns.append(len(z))
 q=pd.Series(vals,index=pd.to_datetime(dates)); print('h',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 for lab,a,b in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025-26','2025','2026')]:
  z=q.loc[a:b];print(lab,len(z),z.mean(),z.mean()/z.std(ddof=1))
 # decay 10d
 if h=='r1':
  for k in [3,5,10]:
   v=[]
   for dt in all_dates:
    z=pd.DataFrame({'x':[D[s].at[dt,'sig'] if dt in D[s].index else np.nan for s in U], 'y':[D[s].close.pct_change(k).shift(-k).get(dt,np.nan) if dt in D[s].index else np.nan for s in U]},index=U).dropna()
    if len(z)>=8 and z.x.nunique()>1:v.append(spearmanr(z.x,z.y).statistic)
   v=np.array(v);print('decay',k,len(v),v.mean(),v.mean()/v.std(ddof=1))
# save signal artifact
S.rename_axis('date').reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20261105_volume_shock_reversal_signal.csv',index=False)

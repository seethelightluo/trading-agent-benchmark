import pandas as pd, numpy as np, glob, os
files={
'vol5':'../persistent/factor_signals_miner_1_20270225_volconfirm_reversal5.csv',
'vixdisp':'../persistent/factor_signals_miner_3_20270226_vix_dispersion_reversal3.csv',
'tail':'../persistent/factor_signals_miner_2_20270225_tail_rebound.csv'}
xs={}
for k,f in files.items():
 d=pd.read_csv(f)
 # normalize wide/long artifacts
 if 'asset' in d.columns: a='asset'
 elif 'symbol' in d.columns: a='symbol'
 elif 'Unnamed: 1' in d.columns: a='Unnamed: 1'
 else:
  d=d.melt(id_vars=['date'],var_name='asset',value_name='signal'); a='asset'
 if 'signal' not in d: continue
 d['date']=pd.to_datetime(d.date); d[a]=d[a].astype(str)
 xs[k]=d.set_index(['date',a]).signal.rename(k)
allx=pd.concat(xs.values(),axis=1)
print('artifacts', {k:len(v) for k,v in xs.items()}, 'merged',allx.shape)
print(allx.corr(min_periods=100).round(4).to_string())
print('pairwise valid', allx.notna().sum().to_dict())
# date-wise cross-sectional IC proxy correlations between factors
for i in xs:
 for j in xs:
  if i<j:
   z=allx[[i,j]].dropna(); print(i,j,'rho',z[i].corr(z[j]),'n',len(z))

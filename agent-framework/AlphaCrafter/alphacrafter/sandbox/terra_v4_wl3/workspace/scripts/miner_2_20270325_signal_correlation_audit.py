import pandas as pd
import numpy as np
from pathlib import Path
files={
'yieldspread_residual':'scripts/miner_2_20270325_yieldspread_residual_signal.csv',
'plain_reversal_2d':'scripts/miner_2_20270325_plain_reversal_2d_signal.csv',
'volatility_surprise':'scripts/miner_2_20270325_volatility_surprise_reversal_signal.csv',
'stress_rebound':'scripts/miner_3_20270326_stress_rebound_signal.csv',
}
xs={}
for name,fn in files.items():
 d=pd.read_csv(fn).set_index('date')
 # cross-sectional daily signal vector, demean and standardize each date;
 # flatten only dates with >=8 common valid names for pair
 xs[name]=d
 print(name, 'dates',len(d),'assets',len(d.columns),'nonempty',int(d.notna().sum().sum()))
for a in files:
 for b in files:
  if a>=b: continue
  A,B=xs[a].align(xs[b],join='inner',axis=0)
  vals=[]; dates=[]
  for dt in A.index:
   z=pd.concat([A.loc[dt].rename('a'),B.loc[dt].rename('b')],axis=1).dropna()
   if len(z)>=8 and z.a.std()>0 and z.b.std()>0:
    vals.append(z.a.corr(z.b)); dates.append(dt)
  print(f'PAIR {a} vs {b}: n_dates={len(vals)} mean_rho={np.mean(vals):.6f} abs_mean={abs(np.mean(vals)):.6f} median={np.median(vals):.6f} date_mean_abs={np.mean(np.abs(vals)):.6f}')
# full flattened correlation as audit backup
for a in files:
 for b in files:
  if a>=b: continue
  A,B=xs[a].align(xs[b],join='inner',axis=0)
  q=pd.concat([A.stack().rename('a'),B.stack().rename('b')],axis=1).dropna()
  print(f'FLAT {a} vs {b}: n={len(q)} rho={q.a.corr(q.b):.6f}')

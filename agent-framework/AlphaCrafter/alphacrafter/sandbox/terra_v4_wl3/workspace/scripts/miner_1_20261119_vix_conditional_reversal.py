import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); macro=Path('../persistent/index_data')
def load(p):
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); return d.drop_duplicates('date').set_index('date')['close'].astype(float).sort_index()
p={s:load(base/(s+'.csv')) for s in U}; v=load(macro/'VIX.csv')
# Use only information available at date t; macro shock is trailing 5 observed VIX sessions.
shock=v.pct_change(fill_method=None).rolling(5,min_periods=5).sum()
pos=shock.clip(lower=0)
# Conditional amplification remains defined on all macro-valid dates, avoiding all-panel dropna.
f={}
for s,x in p.items():
 r=x.pct_change(fill_method=None).rolling(5,min_periods=5).sum()
 f[s]=(-r*(1+pos.reindex(x.index).fillna(0))).rename(s)
f=pd.DataFrame(f).sort_index()
# forward one completed observation for each asset (not calendar shift)
fr={s:(x.shift(-1)/x-1).rename(s) for s,x in p.items()}; fr=pd.DataFrame(fr).sort_index()
rows=[]; sig=[]
for d in f.index:
 a=f.loc[d].dropna(); b=fr.loc[d].reindex(a.index).dropna(); a=a.reindex(b.index)
 if len(a)>=8:
  ic=spearmanr(a,b).statistic
  rows.append((d,ic,len(a))); sig.append((d,a))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
print('IC %.5f ICIR %.5f hit %.3f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
for a,b in [('2020-2022','2022-12-31'),('2023-2024','2024-12-31'),('2025-2026','2026-12-31')]:
 st={'2020-2022':'2020-01-01','2023-2024':'2023-01-01','2025-2026':'2025-01-01'}[a]; z=r.loc[st:b].ic; print(a,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
# turnover using ranked cross-sectional signal changes, available consecutive dates
q=f.rank(axis=1,pct=True); print('turnover',q.diff().abs().mean(axis=1).mean())
out=pd.DataFrame(index=f.index,columns=U); out.loc[:,:]=np.nan
for d,a in sig:
 out.loc[d,a.index]=a
out.index.name='date'; out.to_csv('scripts/miner_1_20261119_vix_conditional_reversal_signal.csv')

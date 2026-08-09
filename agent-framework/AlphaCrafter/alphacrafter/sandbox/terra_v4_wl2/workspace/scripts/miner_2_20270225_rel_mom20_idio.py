import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
px={}
for f in Path('../persistent/stock_data').glob('*.csv'):
 d=pd.read_csv(f,parse_dates=['date']).set_index('date'); px[f.stem]=d.close
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); m=r.median(axis=1)
raw=p.pct_change(20).sub(p.pct_change(20).median(axis=1),axis=0)
idio=r.sub(m,axis=0).rolling(20).std(); fac=raw/idio
ics=[]; dates=[]; ns=[]; turns=[]
for i in range(21,len(p)-1):
 x=fac.iloc[i]; y=r.iloc[i+1]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z): ics.append(z); dates.append(p.index[i]); ns.append(ok.sum())
  if i>21:
   pr=fac.iloc[i-1]; oo=ok&pr.notna(); turns.append((x[oo].rank()-pr[oo].rank()).abs().mean()/max(1,oo.sum()))
a=np.array(ics); dt=pd.DatetimeIndex(dates)
print('dates',len(a),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
print('coverage',fac.notna().mean().mean(),'turn',np.mean(turns))
for start,end in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2027-12-31')]:
 q=a[(dt>=start)&(dt<=end)];print(start,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=pd.DataFrame(fac.loc[dates].values,index=dates,columns=fac.columns);out.index.name='date';out.reset_index().to_csv('../persistent/factor_signals_miner_2_20270225_rel_mom20_idio.csv',index=False)

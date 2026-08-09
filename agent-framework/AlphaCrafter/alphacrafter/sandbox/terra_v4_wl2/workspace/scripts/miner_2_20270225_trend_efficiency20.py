import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
root=Path('../persistent/stock_data')
files=list(root.glob('*.csv'))
px={}
for f in files:
 d=pd.read_csv(f,parse_dates=['date']).set_index('date')
 if 'close' in d: px[f.stem]=d.close
p=pd.DataFrame(px).sort_index()
r=p.pct_change()
# trend efficiency: directional displacement / path length, signed
fac=(p.pct_change(20))/(r.abs().rolling(20).sum())
# require broad data
rows=[]; ic=[]; ns=[]; turns=[]
for i in range(20,len(p)-1):
 x=fac.iloc[i]; y=r.iloc[i+1]
 ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  ic.append(z); ns.append(ok.sum()); rows.append((p.index[i],*x.tolist()))
  if i>20:
   prev=fac.iloc[i-1]; oo=ok&(prev.notna())
   turns.append((x[oo].rank().sub(prev[oo].rank()).abs().mean()/max(1,oo.sum())))
a=np.array(ic); print('dates',len(a),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)))
print('coverage',fac.notna().mean().mean(),'turn',np.mean(turns))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
 q=pd.Series(ic,index=[x[0] for x in rows]).loc[lo:hi]; print(lo,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
# save artifact wide with date and signals
out=pd.DataFrame(rows,columns=['date']+list(p.columns)); out.to_csv('../persistent/factor_signals_miner_2_20270225_trend_efficiency20.csv',index=False)

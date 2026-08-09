import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-11-18')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date').close for s in U}
p=pd.DataFrame(D).sort_index(); R=p.pct_change(); bench=R.median(axis=1); resid=R.sub(bench,axis=0); Y={h:p.shift(-h)/p-1 for h in [1,5,10]}
for w in [2,3,5,10,20]:
 F=-resid.rolling(w,min_periods=w).sum(); out={}
 for h,y in Y.items():
  vals=[]
  for dt in F.index:
   g=pd.DataFrame({'f':F.loc[dt],'y':y.loc[dt]}).dropna()
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
  a=pd.DataFrame(vals,columns=['date','ic','n']); z=a.ic.to_numpy(); out[h]=(len(a),a.n.mean(),z.mean(),z.mean()/z.std(ddof=1),(z>0).mean())
 ranks=F.rank(axis=1,pct=True); turn=ranks.diff().abs().mean().mean()
 print('w',w,'turn',round(turn,4),'coverage',round(F.notna().sum().sum()/F.size,4),out)
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=[]
  for h,y in Y.items():
   for dt in F.index:
    if lo<=dt.year<=hi:
     g=pd.DataFrame({'f':F.loc[dt],'y':y.loc[dt]}).dropna()
     if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
  q=np.array(q); print(' regime',lo,hi,'pooled horizons',len(q),'ICIR',q.mean()/q.std(ddof=1))

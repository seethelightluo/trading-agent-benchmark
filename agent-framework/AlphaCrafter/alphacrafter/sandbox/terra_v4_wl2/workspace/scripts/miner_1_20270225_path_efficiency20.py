import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U if os.path.exists('../persistent/stock_data/'+s+'.csv')}
px=pd.DataFrame(P).sort_index(); r=px.pct_change();
for w in [10,20,30,60]:
 f=r.rolling(w,min_periods=w).sum()/r.abs().rolling(w,min_periods=w).sum()
 for h in [1,5,10]:
  y=px.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]; ranks=[]
  for dt in f.index:
   g=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
    vals.append(spearmanr(g.f,g.y).statistic); dates.append(dt); ns.append(len(g)); ranks.append(g.f.rank(pct=True))
  z=np.array(vals); print('w',w,'h',h,'dates',len(z),'avg_n',np.mean(ns),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turn',pd.DataFrame(ranks).diff().abs().mean().mean())
  for name,mask in [('2020-22',[d.year<=2022 for d in dates]),('2023-24',[2023<=d.year<=2024 for d in dates]),('2025-26',[2025<=d.year<=2026 for d in dates]),('2027',[d.year==2027 for d in dates])]:
   q=z[mask]; print(name,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# save best exploratory artifact
f=r.rolling(20,min_periods=20).sum()/r.abs().rolling(20,min_periods=20).sum(); f.to_csv('../persistent/factor_signals_miner_1_20270225_path_efficiency20.csv')

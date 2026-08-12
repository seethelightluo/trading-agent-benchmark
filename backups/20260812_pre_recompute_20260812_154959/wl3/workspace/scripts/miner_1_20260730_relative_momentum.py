import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date <= @cut').set_index('date').sort_index() for s in U}
for lb in [10,20,40,60]:
 rows=[]
 for s,x in D.items(): rows.append(pd.DataFrame({'date':x.index,'r':x.close.pct_change(lb),'y':x.close.shift(-1)/x.close-1,'symbol':s}).reset_index(drop=True))
 a=pd.concat(rows,ignore_index=True); a['med']=a.groupby('date').r.transform('median');a['f']=a.r-a.med;a=a.dropna(subset=['f','y']);o=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:o.append(spearmanr(g.f,g.y).statistic)
 o=pd.Series(o);rank=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='symbol',values='rank')
 print(lb,len(o),o.mean(),o.mean()/o.std(ddof=1),(o>0).mean(),rank.diff().abs().mean(axis=1).mean())

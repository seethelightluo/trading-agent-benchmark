import pandas as pd,numpy as np
for look in [2,3]:
 q=pd.read_csv(f'scripts/miner_3_20281130_volscaled_reversal{look}_signal.csv',parse_dates=['date']);a=[]
 for dt,g in q.groupby('date'):
  if len(g)>=8:a.append((dt,g.factor.corr(g.forward_return_1d,method='spearman')))
 a=pd.Series(dict(a)).dropna();print('look',look)
 for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2027-12-31'),('2028','2028-12-31'),('2028-08-01','2028-11-29')]:
  z=a[(a.index>=lo)&(a.index<=hi)];print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),5))
 print('turnover proxy',q.sort_values(['symbol','date']).groupby('symbol').factor.apply(lambda x:(x.rank(pct=True).diff().abs()>0.1).mean()).mean())

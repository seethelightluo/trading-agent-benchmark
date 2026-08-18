import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
# Trend efficiency: direction and persistence of 20d move, normalized by path length
rows=[]
for s,d in D.items():
 c=d.close; r=c.pct_change();
 eff=c.pct_change(20)/(r.abs().rolling(20).sum()+1e-12)
 # winsorization is cross-sectional rank robust; raw signal
 f=eff
 for date in f.index:
  j=d.index.get_loc(date)
  if j+5>=len(d): continue
  fr=c.iloc[j+1:j+6].iloc[-1]/c.iloc[j]-1
  rows.append((date,s,f.loc[date],fr))
x=pd.DataFrame(rows,columns=['date','symbol','f','fr']).dropna()
ics=[]; by=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:
  ics.append(spearmanr(g.f,g.fr).statistic); by.append((dt,ics[-1],len(g)))
a=np.array(ics)
print('dates',len(a),'mean_n',np.mean([z[2] for z in by]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
 q=[v for dt,v,n in by if str(dt)>=lo and str(dt)<=hi]
 print(lo, len(q), np.mean(q) if q else None)
# turnover of top/bottom ranked signal across consecutive available dates
print('coverage',x.groupby('date').size().mean()/15)
print('recent',[(str(dt.date()),round(v,3)) for dt,v,n in by[-10:]])

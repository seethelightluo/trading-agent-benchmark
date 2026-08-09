import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load():
 out={}
 for s in U:
  
  try: d=get_index_daily_data(s, days=5000)
  except FileNotFoundError: d=get_stock_daily_data(s, days=5000)
  if d is not None and len(d):
   d=d.copy(); d['date']=pd.to_datetime(d.date); out[s]=d.set_index('date')
 return out

def spearman(a,b): return pd.Series(a).corr(pd.Series(b),method='spearman')
D=load(); dates=sorted(set.intersection(*[set(x.index) for x in D.values()]))
# Candidate: 20d return residualized against cross-sectional median, scaled by own 20d vol (risk-adjusted relative strength)
rows=[]
for i in range(30,len(dates)-10):
 t=dates[i]; nxt=dates[i+1]
 vals=[]; fw=[]
 for s,d in D.items():
  if t not in d.index or dates[i-20] not in d.index: continue
  p=d.close
  r=p.loc[t]/p.loc[dates[i-20]]-1
  vol=np.log(p).diff().loc[:t].tail(20).std()
  if pd.notna(r) and pd.notna(vol) and vol>0 and nxt in d.index:
   vals.append(r/vol); fw.append(d.close.loc[nxt]/p.loc[t]-1)
 if len(vals)>=8: rows.append((t,spearman(vals,fw),len(vals)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate risk-adjusted relative strength; dates',len(x),'avgN',x.n.mean())
print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027-02')]:
 z=x.loc[a:b].ic
 print(a,b,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
# decay 5/10
for h in [5,10]:
 rows=[]
 for i in range(30,len(dates)-h):
  t=dates[i]; end=dates[i+h]; vals=[]; fw=[]
  for s,d in D.items():
   if t not in d.index or dates[i-20] not in d.index or end not in d.index: continue
   p=d.close; r=p.loc[t]/p.loc[dates[i-20]]-1; vol=np.log(p).diff().loc[:t].tail(20).std()
   if pd.notna(r) and pd.notna(vol) and vol>0: vals.append(r/vol); fw.append(p.loc[end]/p.loc[t]-1)
  if len(vals)>=8: rows.append(spearman(vals,fw))
 z=pd.Series(rows);print('horizon',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1))

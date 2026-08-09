import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2026-07-15'); D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cutoff').sort_values('date').set_index('date'); D[s]=x
# Downside-asymmetry resilience: positive-return participation relative to negative tail burden.
# Higher score means fewer/lower downside shocks over trailing window, with magnitude-normalized downside.
for w in [20,40,60]:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change(); pos=r.clip(lower=0).rolling(w,min_periods=w).mean(); neg=(-r.clip(upper=0)).rolling(w,min_periods=w).mean(); f=pos/(neg+1e-8); y=x.close.shift(-1)/x.close-1
  for dt in x.index:
   if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
 a=pd.DataFrame(rows,columns=['date','s','f','y']); z=[]; ns=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: z.append(spearmanr(g.f,g.y).statistic); ns.append(len(g))
 z=np.array(z); print('w',w,'dates',len(z),'avg_names',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0)))
 rr=a.assign(rk=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rk'); print('turnover',rr.diff().abs().mean().mean(),'coverage',a.s.nunique()/15)
 for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-07-15')]:
  q=[]
  for dt,g in a.groupby('date'):
   if lo<=str(dt.date())<=hi and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:q.append(spearmanr(g.f,g.y).statistic)
  q=np.array(q); print(lo[:4],len(q),'ICIR',q.mean()/q.std(ddof=1),'mean',q.mean())
 for h in [5,10]:
  # forward compounded return
  yy=[]
  for s,x in D.items():
   y=(x.close.shift(-h)/x.close-1); ff=x.close.pct_change().clip(lower=0).rolling(w,min_periods=w).mean()/( (-x.close.pct_change().clip(upper=0)).rolling(w,min_periods=w).mean()+1e-8)
   yy += [(dt,s,float(ff.loc[dt]),float(y.loc[dt])) for dt in x.index if pd.notna(ff.loc[dt]) and pd.notna(y.loc[dt])]
  b=pd.DataFrame(yy,columns=['date','s','f','y']); q=[spearmanr(g.f,g.y).statistic for _,g in b.groupby('date') if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1];q=np.array(q);print('h',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))

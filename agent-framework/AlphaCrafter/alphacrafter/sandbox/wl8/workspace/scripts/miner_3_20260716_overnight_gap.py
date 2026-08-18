import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); rows=[]
for s in U:
 d=pd.read_csv(base/(s+'.csv')); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date')
 d['f']=-(d['open']/d['close'].shift(1)-1); d['fwd']=d['close'].shift(-1)/d['close']-1
 d=d[(d.date>='2020-01-01')&(d.date<='2026-07-15')]; rows.append(d[['date','f','fwd']].assign(symbol=s))
x=pd.concat(rows); ics=[]; nms=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['f','fwd'])
 if len(g)>=8: ics.append(spearmanr(g.f,g.fwd).statistic); nms.append(len(g))
a=np.array(ics); print('dates',len(a),'meanN',np.mean(nms),'coverage',len(a)/x.date.nunique(),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'std',np.nanstd(a,ddof=1))
for h in [5,10,20]:
 rr=[]
 for s in U:
  z=pd.read_csv(base/(s+'.csv')); z['date']=pd.to_datetime(z.date); z=z.sort_values('date'); z=z[(z.date>='2020-01-01')&(z.date<='2026-07-15')]; z['f']=-(z.open/z.close.shift(1)-1); z['r']=z.close.shift(-h)/z.close-1; rr.append(z[['date','f','r']])
 q=pd.concat(rr); vals=[]
 for dt,g in q.groupby('date'):
  g=g.dropna()
  if len(g)>=8: vals.append(spearmanr(g.f,g.r).statistic)
 v=np.array(vals); print(h,'IC',np.nanmean(v),'ICIR',np.nanmean(v)/np.nanstd(v,ddof=1),'N',len(v))
z=x.dropna(subset=['f']).copy(); z['rank']=z.groupby('date')['f'].rank(pct=True); z=z.sort_values(['symbol','date']); z['rd']=z.groupby('symbol')['rank'].diff().abs(); print('rank turnover',z.rd.mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 aa=[]
 for dt,g in x.groupby('date'):
  ds=str(dt)[:10]
  if ds>=lo and ds<=hi:
   g=g.dropna()
   if len(g)>=8: aa.append(spearmanr(g.f,g.fwd).statistic)
 print('regime',lo,hi,len(aa),np.nanmean(aa) if aa else np.nan)

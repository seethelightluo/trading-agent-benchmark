import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end].close.pct_change().rename('v')
for win in [20,40,90,120,180]:
 rows=[]
 for s in U:
  r=D[s].close.pct_change().rename('r'); z=pd.concat([r,v],axis=1).dropna()
  f=(-z.r.rolling(win,min_periods=max(15,int(win*.75)).__class__ and max(15,int(win*.75))).cov(z.v)/z.v.rolling(win,min_periods=max(15,int(win*.75))).var()).rename(s)
  rows.append(f)
 F=pd.concat(rows,axis=1).sort_index(); ics=[];ns=[]; years={}
 for dt in F.index:
  x=[];y=[]
  for s in U:
   if dt not in D[s].index or pd.isna(F.loc[dt,s]): continue
   i=D[s].index.get_loc(dt)
   if i+1<len(D[s]) and pd.notna(D[s].close.iloc[i+1]): x.append(F.loc[dt,s]);y.append(D[s].close.iloc[i+1]/D[s].close.iloc[i]-1)
  if len(x)>=8 and len(set(x))>1:
   q=spearmanr(x,y).statistic;ics.append(q);ns.append(len(x));years.setdefault(str(dt.year),[]).append(q)
 a=np.array(ics); print('WIN',win,'N',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'years',{k:round(np.mean(z),4) for k,z in years.items()},'coverage',round(F.notna().sum().sum()/(len(F)*15),4),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()*2,4))
# 5d smoothed 20d beta as a separate candidate diagnostic

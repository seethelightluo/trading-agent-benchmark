import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U},axis=1).sort_index().loc[:'2028-09-06']; r=p.pct_change(); disp=r.std(axis=1).rolling(5).mean(); shock=disp>disp.rolling(60,min_periods=30).quantile(.70); f=-r.rolling(5).sum()
for h in [5,10]:
 out=[]
 for i in range(60,len(p)-h):
  if not shock.iloc[i]: continue
  ok=f.iloc[i].notna()&p.iloc[i+h].notna()
  if ok.sum()>=8: out.append(spearmanr(f.iloc[i][ok],(p.iloc[i+h]/p.iloc[i]-1)[ok]).statistic)
 a=pd.Series(out).dropna(); print('shock only h',h,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
 for lo,hi in [('2020','2025'),('2026','2026'),('2027','2027'),('2028','2028')]:
  dates=p.index[60:len(p)-h]; mask=shock.iloc[60:len(p)-h].values & (dates.year>=int(lo)) & (dates.year<=int(hi)); z=np.array(out) # can't align due filtered
 print('coverage',ok.mean())

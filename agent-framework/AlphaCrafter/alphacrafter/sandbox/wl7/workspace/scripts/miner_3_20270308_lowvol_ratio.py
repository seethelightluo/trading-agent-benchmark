import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); d=d.loc[:'2027-03-08']; r=d.close.pct_change()
  # lagged inverse volatility, with slow volatility trend adjustment
  v=r.rolling(20).std(); slow=r.rolling(60).std()
  data[s]=pd.DataFrame({'f':-(v/slow).shift(1),'r':r.shift(-1)})
dates=sorted(set().union(*[set(x.index) for x in data.values()])); ic=[]; ns=[]; sig=[]
for dt in dates:
 vs=[]; yy=[]
 for s,x in data.items():
  if dt in x.index and pd.notna(x.loc[dt].f) and pd.notna(x.loc[dt].r):vs.append(x.loc[dt].f);yy.append(x.loc[dt].r)
 if len(vs)>=8:ic.append(spearmanr(vs,yy).statistic);ns.append(len(vs))
a=np.array(ic);print('dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'coverage',np.mean(ns)/15)
for h in [5,10,20]:
 z=[]
 for dt in dates:
  vs=[];yy=[]
  for s,x in data.items():
   if dt in x.index:
    f=x.loc[dt,'f']; y=x.r.shift(-1).rolling(h).sum().loc[dt]
    if pd.notna(f) and pd.notna(y):vs.append(f);yy.append(y)
  if len(vs)>=8:z.append(spearmanr(vs,yy).statistic)
 print('h',h,'n',len(z),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1))
out=[]
for dt in dates:
 for s,x in data.items():
  if dt in x.index and pd.notna(x.loc[dt,'f']):out.append({'date':dt,'symbol':s,'signal':x.loc[dt,'f']})
pd.DataFrame(out).to_csv('scripts/miner_3_20270308_lowvol_ratio_signal.csv',index=False)

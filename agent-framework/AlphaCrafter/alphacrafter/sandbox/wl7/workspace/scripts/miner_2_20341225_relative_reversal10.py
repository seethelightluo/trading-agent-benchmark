import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-12-25'); px={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]; d=pd.read_csv(f); d.date=pd.to_datetime(d.date); px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); ret10=p.pct_change(10)
# Relative 10-day reversal, normalized by 30-day annualized volatility, lag one day.
f=(-(ret10-ret10.median(axis=1).values[:,None])).div(r.rolling(30).std()*np.sqrt(252)).shift(1)
for h in [5,10,20]:
 rr=p.pct_change(h).shift(-h); a=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],rr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a); print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 if h==10:print('recent500',a[-500:].mean(),a[-500:].mean()/a[-500:].std(ddof=1))
print('assets',len(p.columns),'dates',len(p),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20341225_relative_reversal10_signal.csv')

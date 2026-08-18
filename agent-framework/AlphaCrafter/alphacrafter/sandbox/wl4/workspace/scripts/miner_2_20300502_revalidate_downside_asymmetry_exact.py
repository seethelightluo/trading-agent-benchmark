import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-05-01'); watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in watch: raw[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
px=pd.concat({s:raw[s].close for s in watch if s in raw},axis=1).sort_index(); r=px.pct_change(); down=r.clip(upper=0).rolling(20,min_periods=15).std(); sig=(-r.rolling(15,min_periods=15).sum()/(down+1e-8)).shift(1); y=px.pct_change().shift(-1)
a=[];ns=[];rr=[];dates=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z));dates.append(dt);rr.append(sig.loc[dt].rank(pct=True))
a=np.array(a); print(f'cutoff={CUT.date()} dates={len(a)} avgN={np.mean(ns):.2f} assets={px.shape[1]} IC={a.mean():.9f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(len(a)):.9f} hit={np.mean(a>0):.6f} turnover={np.mean([(rr[i]-rr[i-1]).abs().mean() for i in range(1,len(rr))]):.9f}')
for st in ['2028-01-01','2029-05-01','2029-11-01','2030-01-01']:
 q=np.array([v for d,v in zip(dates,a) if d>=pd.Timestamp(st)]);print(st,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)))

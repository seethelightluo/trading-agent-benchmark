import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2031-08-06')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in watch: raw[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
px=pd.concat({s:raw[s].close for s in watch if s in raw},axis=1).sort_index(); r=px.pct_change()
vol=r.rolling(20,min_periods=15).std(); dd=px/px.rolling(60,min_periods=40).max()-1
sig=((dd.clip(upper=0).abs())*(r.rolling(10,min_periods=8).sum().clip(lower=0))/(vol+1e-8)).shift(1)
for h in [5,10,20]:
 y=px.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]; ranks=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); dates.append(dt); ns.append(len(z)); ranks.append(sig.loc[dt].rank(pct=True))
 a=np.asarray(vals); ir=a.mean()/(a.std(ddof=1)+1e-12)
 print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} minN={min(ns)} coverage={np.mean(ns)/len(watch):.4f} IC={a.mean():.6f} ICIR={ir:.6f} dailyIR={ir/np.sqrt(len(a)):.6f} hit={np.mean(a>0):.4f} turnover={np.mean([(ranks[i]-ranks[i-1]).abs().mean() for i in range(1,len(ranks))]):.4f}')
 for w in [365,730,1095]:
  q=a[-min(w,len(a)):]; print(f' recent{w} IC={q.mean():.6f} ICIR={q.mean()/(q.std(ddof=1)+1e-12):.6f} dates={len(q)}')

import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
end=pd.Timestamp('2027-09-08')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').sort_index()
C=pd.concat({s:load(s).close for s in A},axis=1).loc[:end].sort_index()
R=C.pct_change(); trend=C.pct_change(40); market_trend=trend.median(axis=1)
rows=[]
for s in A:
 r=R[s]; rel=trend[s]-market_trend
 dd=r.where(r<0,0).rolling(40).std(); breadth=(R[s]>0).rolling(20).mean()
 f=(rel/(dd+0.004))*(0.5+0.5*breadth); f=f.shift(1)
 for h in [5,10,20]:
  fw=C[s].shift(-h)/C[s]-1
  for dt in f.index:
   if pd.notna(f.loc[dt]) and pd.notna(fw.loc[dt]): rows.append((dt,s,float(f.loc[dt]),h,float(fw.loc[dt])))
df=pd.DataFrame(rows,columns=['date','symbol','factor','h','fwd'])
for h,g in df.groupby('h'):
 vals=[]
 for dt,x in g.groupby('date'):
  if len(x)>=8:
   z=spearmanr(x.factor,x.fwd).statistic
   if np.isfinite(z): vals.append(z)
 a=np.array(vals); print('horizon n IC ICIR hit',h,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
p=df[df.h==20].pivot(index='date',columns='symbol',values='factor'); ranks=p.rank(axis=1,pct=True)
print('cutoff',end.date(),'dates',df.date.nunique(),'symbols',df.symbol.nunique(),'coverage',p.notna().mean().mean(),'turnover',ranks.diff().abs().mean().mean())
df[df.h==20].to_csv('scripts/miner_3_20270909_market_relative_stability_signal.csv',index=False)
for label,mask in [('2026+',df.date>='2026-01-01'),('2027',df.date>='2027-01-01'),('Q2+',df.date>='2027-04-01')]:
 g=df[(df.h==20)&mask]; vals=[]
 for dt,x in g.groupby('date'):
  if len(x)>=8: vals.append(spearmanr(x.factor,x.fwd).statistic)
 a=np.array(vals);print(label,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1))

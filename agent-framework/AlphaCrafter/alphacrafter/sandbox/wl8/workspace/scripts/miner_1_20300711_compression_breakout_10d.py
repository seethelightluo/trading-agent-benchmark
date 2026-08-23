import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-07-10'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
vol20=r.rolling(20,min_periods=15).std(); vol60=r.rolling(60,min_periods=40).std()
compression=(vol60/vol20).clip(.5,2.)
f=(p.pct_change(20)/(vol20*np.sqrt(20))*compression.clip(.75,1.5)).replace([np.inf,-np.inf],np.nan)
ics=[];ns=[];cs=[];turn=[];ds=[]
for i in range(len(p)-10):
 if p.index[i]<p.index[70] or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 v=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(v):
  ics.append(v);ns.append(ok.sum());cs.append(ok.mean());ds.append(p.index[i])
  if i>0:
   q=f.iloc[i-1]; oo=x.notna()&q.notna()
   if oo.sum(): turn.append((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
a=np.array(ics); D=pd.DatetimeIndex(ds)
print({'factor':'compression_confirmed_breakout_20d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':round(float(np.mean(ns)),2),'coverage':round(float(np.mean(cs)),5),'ic':round(float(a.mean()),6),'icir':round(float(a.mean()/a.std(ddof=1)),6),'hit':round(float(np.mean(a>0)),5),'turnover':round(float(np.mean(turn)),6)})
for name,m in [('180',D>=pd.Timestamp('2029-12-01')),('360',D>=pd.Timestamp('2029-01-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01'))]:
 z=a[m]; print(name,len(z),round(float(z.mean()),6) if len(z) else None,round(float(z.mean()/z.std(ddof=1)),6) if len(z)>1 else None)
pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_1_20300711_compression_breakout_10d_ic.csv',index=False)
f.loc[D].to_csv('scripts/miner_1_20300711_compression_breakout_10d_signal.csv')

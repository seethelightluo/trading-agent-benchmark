import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2030-09-04'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']; px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Candidate: volatility-normalized trend acceleration: recent 20d return minus slow 60d return, scaled by 20d downside risk.
down=np.sqrt((r.clip(upper=0)**2).rolling(20,min_periods=10).mean())
f=(p.pct_change(20)-p.pct_change(60)).div(down,axis=0)
ics=[];ns=[];cs=[];turns=[];dates=[];signals=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-07-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 v=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(v):
  ics.append(v);ns.append(ok.sum());cs.append(ok.mean());dates.append(p.index[i]);signals.append(x)
  if len(signals)>1:
   q=signals[-2];oo=x.notna()&q.notna();turns.append((x[oo].rank(pct=True)-q[oo].rank(pct=True)).abs().mean())
a=np.array(ics);D=pd.DatetimeIndex(dates)
print({'factor':'downside_vol_trend_acceleration_20_60','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cs)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turns))})
for name,m in [('180',D>=pd.Timestamp('2030-03-01')),('360',D>=pd.Timestamp('2029-09-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01')),('recent',D>=pd.Timestamp('2030-04-01'))]:
 z=a[m];print(name,len(z),float(z.mean()) if len(z) else None,float(z.mean()/z.std(ddof=1)) if len(z)>1 else None)
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_1_20300905_downside_vol_trend_acceleration_20_60_ic.csv',index=False)
pd.DataFrame(signals,index=dates,columns=U).to_csv('scripts/miner_1_20300905_downside_vol_trend_acceleration_20_60_signal.csv')

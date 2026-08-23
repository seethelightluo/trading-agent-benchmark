import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-11-13')
cl={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U};p=pd.DataFrame(cl).sort_index();r=p.pct_change();
# Volatility regime reversal: reverse 10d return, weighted by recent 20d volatility rank (extremes more actionable).
ret=p.pct_change(10); vol=r.rolling(20).std(); vr=vol.rank(axis=1,pct=True); f=-ret*(0.5+vr)
dates=[];a=[];ns=[];cov=[];turn=[];prev=None;sig=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut:continue
 x=f.iloc[i];y=p.iloc[i+10]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  dates.append(p.index[i]);a.append(z);ns.append(ok.sum());cov.append(ok.mean());sig.append(x.values)
  if prev is not None:
   oo=x.notna()&prev.notna();turn.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(dates);a=np.array(a)
def out(z):return len(z),float(z.mean()),float(z.mean()/z.std(ddof=1))
print({'factor':'volatility_rank_reversal_10d','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(a.mean()),'icir':float(a.mean()/a.std(ddof=1)),'hit':float((a>0).mean()),'turnover':float(np.mean(turn))})
for n,m in [('180',D>=pd.Timestamp('2030-05-01')),('360',D>=pd.Timestamp('2029-11-01')),('2029',(D>=pd.Timestamp('2029-01-01'))&(D<pd.Timestamp('2030-01-01'))),('2030',D>=pd.Timestamp('2030-01-01')),('recent60',D>=pd.Timestamp('2030-08-01'))]:print(n,*out(a[m]))
pd.DataFrame({'date':dates,'ic':a}).to_csv('scripts/miner_1_20301114_volatility_rank_reversal_10d_ic.csv',index=False)
pd.DataFrame(sig,index=dates,columns=U).to_csv('scripts/miner_1_20301114_volatility_rank_reversal_10d_signal.csv')

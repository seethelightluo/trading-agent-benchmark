import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2031-10-30')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index();r=p.pct_change();f=-r.rolling(5).sum().shift(1)/(r.rolling(20).std().shift(1).clip(lower=.004)*np.sqrt(5));D=[];a=[];ns=[];cov=[];turn=[];prev=None
for i in range(len(p)-5):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+5]>cut:continue
 x=f.iloc[i];y=p.iloc[i+5]/p.iloc[i]-1;ok=x.notna()&y.notna()
 if ok.sum()<8:continue
 z=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(z):
  D.append(p.index[i]);a.append(z);ns.append(ok.sum());cov.append(ok.mean())
  if prev is not None:
   oo=x.notna()&prev.notna();turn.append((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean())
  prev=x
D=pd.DatetimeIndex(D);a=np.array(a);print({'factor':'vol_scaled_shock_reversal_5d_forward5','dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':float(np.mean(ns)),'coverage':float(np.mean(cov)),'ic':float(np.mean(a)),'icir':float(np.mean(a)/np.std(a,ddof=1)),'hit':float(np.mean(a>0)),'turnover':float(np.mean(turn))})
for lab,m in [('365',D>=pd.Timestamp('2030-10-30')),('180',D>=pd.Timestamp('2031-04-30')),('60',D>=pd.Timestamp('2031-08-01'))]:
 z=a[m];print(lab,len(z),float(np.mean(z)),float(np.mean(z)/np.std(z,ddof=1)))
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_1_20311030_shock_reversal_5f_signal.csv');pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_1_20311030_shock_reversal_5f_ic.csv',index=False)

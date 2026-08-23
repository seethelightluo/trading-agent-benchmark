import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2030-12-25')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index(); r=p.pct_change()
# Downside-adjusted short-term reversal: penalize assets with unstable negative-return tails.
down=np.sqrt((r.clip(upper=0)**2).rolling(30).mean())
f=-p.pct_change(10)/down
D=[]; A=[]; N=[]; C=[]; T=[]; prev=None
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2020-06-01') or p.index[i+10]>cut: continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()<8: continue
 q=spearmanr(x[ok],y[ok]).statistic
 if np.isfinite(q):
  D.append(p.index[i]); A.append(q); N.append(int(ok.sum())); C.append(float(ok.mean()))
  if prev is not None:
   oo=x.notna()&prev.notna(); T.append(float((x[oo].rank(pct=True)-prev[oo].rank(pct=True)).abs().mean()))
  prev=x
D=pd.DatetimeIndex(D); a=np.array(A)
print({'dates':len(a),'start':str(D[0].date()),'end':str(D[-1].date()),'avg_instruments':np.mean(N),'coverage':np.mean(C),'ic':np.mean(a),'icir':np.mean(a)/np.std(a,ddof=1),'hit':np.mean(a>0),'turnover':np.mean(T)})
for n,m in [('180',D>=pd.Timestamp('2030-06-01')),('360',D>=pd.Timestamp('2029-12-01')),('2030',D>=pd.Timestamp('2030-01-01')),('recent60',D>=pd.Timestamp('2030-09-15'))]:
 z=a[m]; print(n,len(z),np.mean(z),np.mean(z)/np.std(z,ddof=1) if len(z)>1 else np.nan)
pd.DataFrame({'date':D,'ic':a}).to_csv('scripts/miner_2_20301226_downside_reversal_ic.csv',index=False)
pd.DataFrame([f.loc[d].values for d in D],index=D,columns=U).to_csv('scripts/miner_2_20301226_downside_reversal_signal.csv')

import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; d={}
for a in A:
 f=f'{b}/{a}.csv'
 if os.path.exists(f): d[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
D=pd.concat(d,axis=1).sort_index(); C=D.xs('close',axis=1,level=1); r=C.pct_change()
# Idea: recent losers are favored, but normalize by downside volatility to avoid treating noisy assets as attractive
lb=10; down=r.where(r<0).rolling(30,min_periods=15).std(); raw=-(C/C.shift(lb)-1); F=raw/down
ics_by={h:[] for h in [1,5,10,20]}; nobs=[]; turns=[]; prev=None
for t in F.index:
 x=F.loc[t]; nobs.append((x.notna()).sum())
 y=C.shift(-1).loc[t]/C.loc[t]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  q=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(q): ics_by[1].append(q)
  rr=x[ok].rank(pct=True)
  if prev is not None: turns.append((rr-prev.reindex(rr.index)).abs().mean())
  prev=rr
 for h in [5,10,20]:
  y=C.shift(-h).loc[t]/C.loc[t]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   q=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(q): ics_by[h].append(q)
print('idea downside_adjusted_reversal_10d; dates',len(F.index),'universe',len(d),'avg_valid',np.mean(nobs),'coverage',np.mean(np.array(nobs)>=8))
for h,v in ics_by.items():
 s=pd.Series(v); print('horizon',h,'n',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
print('turnover',np.mean(turns),'latest',F.iloc[-1].dropna().round(3).to_dict())

import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-07-29'
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index(); D[s]=x.close
p=pd.DataFrame(D).sort_index(); r=p.pct_change();
# medium-term trend persistence: trailing 20-day return, risk normalized by trailing 20-day realized volatility
f=r.rolling(20,min_periods=15).sum()/(r.rolling(20,min_periods=15).std()*np.sqrt(20)+1e-12)
y=p.pct_change().shift(-1)
ics=[]; ns=[]
for d in f.index:
 a=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
 if len(a)>=8 and a.f.nunique()>1:
  q=spearmanr(a.f,a.y).statistic
  if pd.notna(q):ics.append((d,q));ns.append(len(a))
z=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); q=z.ic
print('candidate trend_persistence_20d_volnorm cutoff',cut,'dates',len(q),'avg_n',np.mean(ns),'coverage',f.notna().sum().sum()/f.size,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-29')]:
 a=z.loc[lo:hi].ic;print('regime',lo,hi,len(a),a.mean(),a.mean()/a.std(ddof=1))
for h in [3,5,10]:
 yy=p.pct_change(h).shift(-h);v=[]
 for d in f.index:
  a=pd.DataFrame({'f':f.loc[d],'y':yy.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1:v.append(spearmanr(a.f,a.y).statistic)
 print('decay',h,len(v),np.mean(v),np.mean(v)/np.std(v,ddof=1))
print('period',p.index.min(),p.index.max())

import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-11-17')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); sig=(r.rolling(20,min_periods=18).sum()/r.rolling(20,min_periods=18).std()).shift(0); fwd=px.shift(-1)/px-1
ics=[]; ns=[]; dates=[]; turnover=[]; prev=None
for d in sig.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): ics.append(q);ns.append(len(g));dates.append(d)
 ranks=sig.loc[d].rank(pct=True)
 if prev is not None: turnover.append((ranks-prev).abs().mean())
 prev=ranks
arr=np.array(ics); dates=pd.to_datetime(dates)
def calc(mask):
 a=arr[mask]; return len(a), round(float(np.mean(a)),6), round(float(np.mean(a)/np.std(a,ddof=1)),6), round(float(np.mean(a>0)),4)
print('dates',len(arr),'avg_names',round(np.mean(ns),2),'coverage',round(len(arr)/len(sig),4),'turnover',round(float(np.nanmean(turnover)),4)); print('overall',calc(np.ones(len(arr),bool)))
for q,m in [('2020-22',(dates<'2023-01-01')),('2023-25',((dates>='2023-01-01')&(dates<'2026-01-01'))),('2026',((dates>='2026-01-01')&(dates<'2027-01-01'))),('2027',(dates>='2027-01-01')),('recent180',(dates>=END-pd.Timedelta(days=180)))]: print(q,calc(m))
for h in [1,3,5,10]:
 yy=(1+r).rolling(h).apply(np.prod,raw=True).shift(-h+1)-1; aa=[]
 for d in sig.index:
  g=pd.DataFrame({'s':sig.loc[d],'f':yy.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1: aa.append(spearmanr(g.s,g.f).statistic)
 aa=np.array(aa); print('horizon',h,'n',len(aa),'IC',round(float(np.nanmean(aa)),6),'ICIR',round(float(np.nanmean(aa)/np.nanstd(aa,ddof=1)),6))
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20271118_vol_adjusted_momentum_signal.csv',index=False)

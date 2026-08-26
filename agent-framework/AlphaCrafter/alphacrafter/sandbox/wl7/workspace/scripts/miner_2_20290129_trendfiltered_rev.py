import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2029-01-28'); P={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  d=d[['date','close']].copy(); d.date=pd.to_datetime(d.date).dt.normalize(); P[s]=d.drop_duplicates('date').set_index('date').close.loc[:CUT]
P=pd.DataFrame(P).sort_index(); r=P.pct_change(); vol=r.rolling(30,min_periods=20).std()
# Trend-filtered dispersion reversal: risk-normalized short reversal is only active
# when the asset's 20d trend is not strongly adverse, reducing falling-knife exposure.
disp=r.rolling(10,min_periods=10).std().mean(axis=1); threshold=disp.rolling(90,min_periods=45).quantile(.60); gate=disp>threshold
raw=-(P.pct_change(5)/(vol*np.sqrt(5))); trend=P.pct_change(20)
# retain reversal in cross-sectional top 2/3 trend, then demean
rank=trend.rank(axis=1,pct=True); f=raw.where(gate,0).where(rank>=.333,0); f=f.sub(f.mean(axis=1),axis=0)
ics=[]; ns=[]
for i in range(len(P)-20):
 x=f.iloc[i]; y=P.iloc[i+20]/P.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].nunique()>1:
  z=x[ok].corr(y[ok],method='spearman')
  if np.isfinite(z): ics.append((P.index[i],z)); ns.append(ok.sum())
q=pd.DataFrame(ics,columns=['date','ic']).set_index('date'); z=q.ic
turn=[]
for i in range(1,len(f)):
 a=f.iloc[i-1].rank(pct=True); b=f.iloc[i].rank(pct=True); ok=a.notna()&b.notna()
 if ok.sum()>=8: turn.append(abs(a[ok]-b[ok]).mean())
print('valid_dates',len(q),'avg_instruments',np.mean(ns),'coverage',np.mean(ns)/15,'active_frac',np.mean(gate),'IC',z.mean(),'ICIR_daily',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.mean(turn))
for a,b in [(2020,2022),(2023,2024),(2025,2026),(2027,2028),(2028,2029),(2029,2029)]:
 w=z[(z.index.year>=a)&(z.index.year<=b)]; print('regime',a,b,'dates',len(w),'IC',w.mean(),'ICIR',w.mean()/w.std(ddof=1) if len(w)>1 else np.nan)
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol',0:'signal'}).to_csv('scripts/miner_2_20290129_trendfiltered_rev_signal.csv',index=False)
# decay diagnostics
for H in [10,20,30]:
 a=[]
 for i in range(len(P)-H):
  x=f.iloc[i]; y=P.iloc[i+H]/P.iloc[i]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8 and x[ok].nunique()>1:a.append(x[ok].corr(y[ok],method='spearman'))
 print('decay_H',H,'dates',len(a),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1))

"""miner_1: VIX-tail conditioned downside-day intraday recovery, trailing 60 native observations."""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-02-20'); HS=(1,5,10,20)
cl={}; op={}; fw={h:{} for h in HS}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
 cl[a]=d.close.astype(float); op[a]=d.open.astype(float)
 r=cl[a].pct_change(fill_method=None)
 for h in HS: fw[h][a]=cl[a].shift(-h)/cl[a]-1
cl=pd.DataFrame(cl).sort_index(); op=pd.DataFrame(op).reindex(cl.index); ret=cl.pct_change(fill_method=None)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
vc=(v['close'] if 'close' in v else v.select_dtypes('number').iloc[:,0]).astype(float).reindex(cl.index)
# A VIX-tail day is a completed day with VIX above its prior 60-observation 80th percentile.
# Signal compares each asset's intraday recovery on tail/down days to its ordinary/down-day recovery.
tail=vc > vc.rolling(60,min_periods=60).quantile(.80).shift(1)
intra=cl/op-1
sig=pd.DataFrame(np.nan,index=cl.index,columns=A)
for t in range(60,len(cl)):
 window=ret.iloc[t-60:t]; path=intra.iloc[t-60:t]; state=tail.iloc[t-60:t]
 cond=(window<0).mul(state,axis=0) # retained only tail-and-down observations
 for a in A:
  good=cond[a].fillna(False); base=(window[a]<0)&(~state.fillna(False))
  if good.sum()>=8 and base.sum()>=12:
   sig.loc[sig.index[t],a]=path.loc[good,a].mean()-path.loc[base,a].mean()
print('FACTOR vix_tail_downside_intraday_recovery_spread_60obs endpoint',END.date(),'assets',len(A))
print('signal cells',int(sig.notna().sum().sum()),'/',sig.size,'coverage',round(float(sig.notna().sum().sum()/sig.size),5),'mean_assets_per_date',round(float(sig.notna().sum(axis=1).mean()),3))
res={}
for h in HS:
 y=pd.DataFrame(fw[h]); vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); dates.append(dt); ns.append(len(q))
 x=np.array(vals); ds=pd.DatetimeIndex(dates); ns=np.array(ns); res[h]=(x,ds,ns)
 print('H',h,'ic_dates',len(x),'mean_IC',round(float(x.mean()),6),'ICIR',round(float(x.mean()/x.std(ddof=1)),6),'hit',round(float((x>0).mean()),5),'mean_instruments',round(float(ns.mean()),3),'min_instruments',int(ns.min()))
x,ds,_=res[10]
for lab,lo,hi in [('2020_21','2020-01-01','2021-12-31'),('2022_23','2022-01-01','2023-12-31'),('2024_25','2024-01-01','2025-12-31'),('2026_30','2026-01-01','2030-02-20')]:
 z=x[(ds>=lo)&(ds<=hi)]; print('REGIME',lab,'dates',len(z),'IC',round(float(z.mean()),6),'ICIR',round(float(z.mean()/z.std(ddof=1)),6),'hit',round(float((z>0).mean()),5))
rk=sig.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(q)>=8: turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('rank_turnover',round(float(np.mean(turn)),6),'turnover_dates',len(turn))
print('ADMISSION_PRECHECK','PASS_IC_GATES' if abs(x.mean())>=.007 and abs(x.mean()/x.std(ddof=1))>=.084 else 'FAIL_IC_GATES')
sig.to_pickle('scripts/miner_1_vix_tail_downside_intraday_recovery_spread_candidate_signal.pkl')

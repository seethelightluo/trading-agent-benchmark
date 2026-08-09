"""miner_1: price-only directional-efficiency persistence conditioned on native range compression.
All features and forward returns are calculated from each asset's own completed bars.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-12-26')
H=(1,5,10,20); F={}; Y={h:{} for h in H}
for a in A:
    d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:END]
    c=d['close'].astype(float); hi=d['high'].astype(float); lo=d['low'].astype(float)
    r=c.pct_change(fill_method=None)
    # Signed 10-bar path efficiency, amplified only when today's range is quiet vs own 20-bar norm.
    # The compression term is clipped; this avoids a single stale/illiquid bar determining ranks.
    eff=(c/c.shift(10)-1).div(r.abs().rolling(10,min_periods=8).sum()).clip(-1,1)
    nr=((hi-lo)/c).replace([np.inf,-np.inf],np.nan)
    compression=np.log(nr.rolling(5,min_periods=4).mean()/nr.rolling(20,min_periods=15).mean()).clip(-1,1)
    F[a]=(eff*(-compression)).replace([np.inf,-np.inf],np.nan)
    for h in H: Y[h][a]=(1+r).rolling(h,min_periods=h).apply(np.prod,raw=True).shift(-h)-1
sig=pd.DataFrame(F); print('FACTOR compressed_range_directional_efficiency_persistence_5_10_20obs cutoff',END.date(),'assets',len(A))
print('cells',int(sig.notna().sum().sum()),'/',sig.size,'coverage',round(sig.notna().sum().sum()/sig.size,6),'mean_assets_per_date',round(sig.notna().sum(axis=1).mean(),3))
def evaluate(h):
    y=pd.DataFrame(Y[h]); ics=[]; dates=[]; counts=[]
    for dt in sig.index:
      q=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
      if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
       ics.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);dates.append(dt);counts.append(len(q))
    x=np.array(ics); return x,pd.DatetimeIndex(dates),np.array(counts)
res={}
for h in H:
 x,d,n=evaluate(h);res[h]=(x,d,n); sd=x.std(ddof=1) if len(x)>1 else np.nan
 print('H',h,'ic_dates',len(x),'daily_paper_IC',round(x.mean(),6),'ICIR',round(x.mean()/sd,6),'hit',round((x>0).mean(),5),'mean_instruments',round(n.mean(),3),'min_instruments',n.min())
# pre-specified 10d operational horizon
x,d,n=res[10]
for lab,lo,hi in [('2020_21','2020-01-01','2021-12-31'),('2022_23','2022-01-01','2023-12-31'),('2024_25','2024-01-01','2025-12-31'),('2026_29','2026-01-01','2029-12-26')]:
 z=x[(d>=lo)&(d<=hi)]; sd=z.std(ddof=1) if len(z)>1 else np.nan
 print('REGIME',lab,'dates',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/sd,6) if len(z)>1 else None,'hit',round((z>0).mean(),5) if len(z) else None)
ranks=sig.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 q=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(q)>=8: turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('rank_turnover',round(float(np.mean(turns)),6),'turnover_dates',len(turns))
sd=x.std(ddof=1); print('ADMISSION_PRECHECK','PASS_IC_GATES' if abs(x.mean())>=.007 and abs(x.mean()/sd)>=.084 else 'FAIL_IC_GATES')
sig.to_pickle('scripts/miner_1_candidate_signal.pkl')

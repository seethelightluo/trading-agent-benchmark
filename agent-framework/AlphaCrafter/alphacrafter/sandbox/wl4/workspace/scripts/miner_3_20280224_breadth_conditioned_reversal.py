import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for a in ASSETS:
    f=f'{base}/{a}.csv'
    if os.path.exists(f):
        d=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
        px[a]=d
P=pd.DataFrame(px).sort_index()
# Observation-only VIX is used only as a regime conditioner, never tradable.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(P.index).ffill()
# candidate: contrarian 5d return, amplified in high-volatility regime using lagged 20d VIX z-score
r=P.pct_change()
vz=(vix-vix.rolling(60,min_periods=40).mean())/vix.rolling(60,min_periods=40).std()
# strictly lag regime by one day; clip prevents excessive conditioning
amp=(1+0.6*vz.shift(1).clip(-1.5,1.5)).clip(0.1,1.9)
factor=-r.rolling(5).sum().shift(1).mul(amp,axis=0)
# forward 5 trading-day return
fwd=P.shift(-5)/P-1
ics=[]; turnovers=[]; ninst=[]
prev=None
for dt in factor.index:
    x=factor.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ic=spearmanr(x[ok],y[ok]).statistic
        if np.isfinite(ic): ics.append((dt,ic)); ninst.append(ok.sum())
        ranks=x[ok].rank(pct=True)
        if prev is not None:
            common=ranks.index.intersection(prev.index)
            turnovers.append((ranks[common]-prev[common]).abs().mean())
        prev=ranks
s=pd.Series(dict(ics)); s.index=pd.to_datetime(s.index)
# report subperiod regimes, and decay
print('dates',len(s),'avg_instruments',round(float(np.mean(ninst)),2),'coverage_pct',round(100*len(s)/(len(factor.index)-5),2))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'turnover',round(float(np.mean(turnovers)),4))
for name,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-28','2026-01-01','2028-02-24')]:
 z=s.loc[lo:hi]; print(name,'n',len(z),'IC',round(z.mean(),6) if len(z) else None,'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>2 else None)
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; q=[]
 for dt in factor.index:
  ok=factor.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(factor.loc[dt][ok],yy.loc[dt][ok]).statistic)
 print('decay',h,round(float(np.nanmean(q)),6),'n',len(q))

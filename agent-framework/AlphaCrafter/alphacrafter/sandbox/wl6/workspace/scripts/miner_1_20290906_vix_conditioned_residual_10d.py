import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
    d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
    px[a]=d
p=pd.DataFrame(px).sort_index()
r=np.log(p).diff()
# Observation-only macro signal, lagged through prior completed day by using rolling history at date t.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill()
# signal at t: residual 10d return, volatility scaled; flip orientation in high-VIX regime
ret10=np.log(p/p.shift(10)); vol30=r.rolling(30).std()*np.sqrt(252)
res=ret10.sub(ret10.mean(axis=1),axis=0)/(vol30+0.02)
reg=(v>v.rolling(60,min_periods=30).median()).astype(float)
f=res*(1-2*reg.values[:,None])
fwd=np.log(p.shift(-10)/p)
ics=[]; dates=[]; nobs=[]
for dt in f.index:
    x=f.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
        z=spearmanr(x[ok],y[ok]).statistic
        if np.isfinite(z): ics.append(z); dates.append(dt); nobs.append(ok.sum())
ics=np.array(ics)
# turnover via rank-normalized cross-sectional signal changes
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('dates',len(ics),'from',dates[0].date(),'to',dates[-1].date(),'avg_n',np.mean(nobs),'coverage',np.mean(nobs)/15)
print('IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1)*np.sqrt(252/10),'hit',np.mean(ics>0),'turnover',turn)
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028'),('2029','2030')]:
 q=ics[(np.array([d.strftime('%Y') for d in dates])>=lo)&(np.array([d.strftime('%Y') for d in dates])<=hi)]
 if len(q): print(lo+'-'+hi,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252/10) if len(q)>1 else np.nan)

import os, numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    p=os.path.join(base,s+'.csv')
    if not os.path.exists(p): continue
    d=pd.read_csv(p)
    d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').set_index('date')
    px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill()
r=P.pct_change()
# signal available at t: risk-adjusted 20d trend, reversed when market breadth is weak
ret20=P/P.shift(20)-1
vol20=r.rolling(20).std()*np.sqrt(252)
raw=ret20/vol20.replace(0,np.nan)
breadth=(r>0).rolling(5).mean().mean(axis=1)
# trailing 60d median is known at t; trend in strong breadth, contrarian in weak breadth
state=(breadth > breadth.rolling(60,min_periods=30).median()).astype(float).replace(0,-1)
f=raw.mul(state,axis=0)
# forward 10-session returns, no look-ahead
fw=P.shift(-10)/P-1
ics=[]; dates=[]; turnovers=[]
prev=None
for dt in f.index:
    a=f.loc[dt]; y=fw.loc[dt]; ok=a.notna()&y.notna()
    if ok.sum()<8: continue
    ic=spearmanr(a[ok],y[ok]).statistic
    if np.isfinite(ic):
      ics.append(ic); dates.append(dt)
      ranks=a[ok].rank(pct=True)
      if prev is not None: turnovers.append(np.abs(ranks-prev.reindex(ranks.index).fillna(.5)).mean())
      prev=ranks
ics=np.array(ics)
print('factor=breadth_conditioned_risk_momentum_20d; dates=%d instruments=%d meanN=%.2f coverage=%.4f'%(len(ics),len(U),np.mean([((f.loc[d].notna()&fw.loc[d].notna()).sum()) for d in dates])/len(U),np.mean([((f.loc[d].notna()&fw.loc[d].notna()).sum()) for d in dates])/len(U)))
for label,mask in [('full',np.ones(len(dates),bool)),('2020-2024',np.array([d.year<=2024 for d in dates])),('2025-2027',np.array([2025<=d.year<=2027 for d in dates])),('2028-2029',np.array([d.year>=2028 for d in dates]))]:
 x=ics[mask]; print(label,'n=%d IC=%.6f ICIR=%.6f hit=%.4f'%(len(x),x.mean(),x.mean()/x.std(ddof=1)*np.sqrt(1) if len(x)>1 and x.std(ddof=1)>0 else np.nan,(x>0).mean()))
print('turnover_mean=%.6f'%np.mean(turnovers))
for h in [5,20,40]:
 yy=P.shift(-h)/P-1; z=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8: z.append(spearmanr(f.loc[dt][ok],yy.loc[dt][ok]).statistic)
 print('horizon',h,'n',len(z),'IC',np.nanmean(z),'ICIR',np.nanmean(z)/np.nanstd(z,ddof=1))

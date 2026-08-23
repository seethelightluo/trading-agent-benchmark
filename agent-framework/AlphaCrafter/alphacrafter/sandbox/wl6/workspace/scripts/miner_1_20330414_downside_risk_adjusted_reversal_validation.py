import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date')
    px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index()
R=P.pct_change()
# Idea: rebound after a severe, downside-vol-adjusted 5d shock; require longer trend context.
# Higher score means stronger rebound candidate, with risk scaled shock and 20d trend confirmation.
down=R.where(R<0).rolling(20,min_periods=10).std()
shock=R.rolling(5,min_periods=5).sum()/((down*np.sqrt(5)).rolling(5,min_periods=5).mean()+1e-8)
trend=R.rolling(20,min_periods=15).sum()
F=(-shock)*(1+0.5*np.tanh(trend/0.10))
# cross-sectional forward 10 trading-day return
fr=P.shift(-10)/P-1
ics=[]; ns=[]; turnovers=[]; dates=[]
prev=None
for dt in F.index:
    x=F.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ic=spearmanr(x[ok],y[ok]).statistic
        if np.isfinite(ic):
            ics.append(ic); ns.append(int(ok.sum())); dates.append(dt)
            rank=x.rank(pct=True)
            if prev is not None: turnovers.append(float((rank-prev).abs().mean()))
            prev=rank
arr=np.array(ics)
print(json.dumps({'factor':'downside_risk_adjusted_reversal_10d','dates':len(arr),'date_start':str(dates[0].date()),'date_end':str(dates[-1].date()),'avg_n':float(np.mean(ns)),'coverage':float(np.mean(ns)/15),'ic':float(np.mean(arr)),'icir':float(np.mean(arr)/(np.std(arr,ddof=1)+1e-12)*np.sqrt(252/10)),'hit':float(np.mean(arr>0)),'turnover':float(np.mean(turnovers))},indent=2))
for h in [5,10,20,40]:
 yy=P.shift(-h)/P-1; aa=[]
 for dt in F.index:
  ok=F.loc[dt].notna()&yy.loc[dt].notna()
  if ok.sum()>=8:
   z=spearmanr(F.loc[dt][ok],yy.loc[dt][ok]).statistic
   if np.isfinite(z): aa.append(z)
 print('decay',h,float(np.mean(aa)),len(aa))
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-12-31'),('2029','2030-12-31'),('2031','2033-04-14')]:
 q=[v for d,v in zip(dates,ics) if pd.Timestamp(a)<=d<=pd.Timestamp(b)]
 print('regime',a,b,len(q),float(np.mean(q)) if q else None,float(np.mean(q)/(np.std(q,ddof=1)+1e-12)*np.sqrt(252/10)) if len(q)>1 else None)
# signal artifact for audit
out=F.reset_index().rename(columns={'date':'timestamp'})
out.to_csv('scripts/miner_1_20330414_downside_risk_adjusted_reversal_signal.csv',index=False)

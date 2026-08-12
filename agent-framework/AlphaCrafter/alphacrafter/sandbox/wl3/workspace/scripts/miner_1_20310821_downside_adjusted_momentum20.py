import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-08-21')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close']
 px[s]=d[d.index<=end]
P=pd.DataFrame(px).sort_index()
r=P.pct_change()
# downside-adjusted medium momentum: 20d return divided by 20d downside deviation, lagged one session
mom=P.shift(1)/P.shift(21)-1
down=r.where(r<0,0).rolling(20,min_periods=12).std().shift(1)
F=mom/(down+1e-8)
# forward 10-session close return from t to t+10
Y=P.shift(-10)/P-1
rows=[]; turnovers=[]; prev=None
for dt in F.index:
 x=F.loc[dt]; y=Y.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
  ranks=x.rank(pct=True).dropna()
  if prev is not None:
   q=pd.concat([prev,ranks],axis=1).dropna(); turnovers.append((q.iloc[:,0]-q.iloc[:,1]).abs().mean())
  prev=ranks
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(res),'avg_n',res.n.mean(),'coverage',res.n.mean()/15)
print('IC',res.ic.mean(),'ICIR',res.ic.mean()/res.ic.std(ddof=1),'hit',(res.ic>0).mean(),'turnover',np.mean(turnovers))
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030'),('2031','2031')]:
 q=res.loc[a:b].ic
 print(a,b,len(q),q.mean() if len(q) else np.nan)
for h in [1,3,5,10,20]:
 yy=P.shift(-h)/P-1; rr=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',np.nanmean(rr),'n',len(rr))
# artifacts
out=F.copy(); out.index=out.index.strftime('%Y-%m-%d'); out.to_csv('scripts/miner_1_20310821_downside_adjusted_momentum20_signal.csv',index_label='date')

import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-06-11')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=end]
P=pd.DataFrame(px).dropna(how='all'); R=P.pct_change()
# adverse breadth shock reversal, all calculations strictly lagged via shift
r5=P.pct_change(5); v20=R.rolling(20).std()*np.sqrt(20)
breadth=(r5>0).mean(axis=1); med=r5.median(axis=1)
active=((breadth<.35)&(med<0)).astype(float)
# two-day persistence, then lag activation one session
act=(active.rolling(2).min().shift(1)>0).astype(float)
f=(-r5/v20).mul(act,axis=0)
# standardize cross-section only for active dates; zeros on inactive are not observations
rows=[]
for i,dt in enumerate(P.index):
 if i+10>=len(P.index): continue
 x=f.loc[dt].replace([np.inf,-np.inf],np.nan); y=P.iloc[i+10]/P.iloc[i]-1
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
 if len(z)>=8 and z['x'].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.y).statistic,len(z),float(act.loc[dt])))
q=pd.DataFrame(rows,columns=['date','ic','n','active']).set_index('date')
for h in [1,3,5,10,20]:
 out=[]
 for i,dt in enumerate(P.index):
  if i+h>=len(P.index):continue
  z=pd.concat([f.loc[dt],(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  z=z[z.iloc[:,0]!=0]
  if len(z)>=8 and z.iloc[:,0].nunique()>1: out.append(spearmanr(z.iloc[:,0],z.y).statistic)
 a=np.array(out); print('H',h,'dates',len(a),'IC',a.mean() if len(a) else np.nan,'ICIR',a.mean()/a.std(ddof=1) if len(a)>1 else np.nan,'hit',(a>0).mean() if len(a) else np.nan)
print('primary all dates',len(q),'active freq',q.active.mean(),'avg n',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028-30','2028','2030-12-31'),('2031','2031','2031-06-11')]:
 z=q.loc[(q.index>=label[:4])&(q.index<=hi)].ic
 print(label,len(z),z.mean() if len(z) else np.nan,z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
print('coverage',q.n.sum()/(len(q)*15),'turnover rough',f.rank(axis=1,pct=True).diff().abs().mean().mean())
# artifact
f.to_csv('scripts/miner_3_20310612_shock_reversal_signal.csv',index_label='date')

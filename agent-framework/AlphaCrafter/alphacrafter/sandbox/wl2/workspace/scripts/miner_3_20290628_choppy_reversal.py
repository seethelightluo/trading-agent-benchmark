import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)==0: d=get_index_daily_data(s,1500)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# 20d cross-asset path efficiency; low efficiency identifies choppy regimes.
eff=(r.rolling(20).sum().abs()/(r.abs().rolling(20).sum()+1e-12)).mean(axis=1)
threshold=eff.shift(1).rolling(500,min_periods=30).quantile(.30)
rows=[]; signals=[]
for t in range(65,len(p)-1):
 if not (np.isfinite(eff.iloc[t]) and np.isfinite(threshold.iloc[t]) and eff.iloc[t]<=threshold.iloc[t]): continue
 f=(-r.iloc[t-4:t+1].sum()/r.iloc[t-19:t+1].std()).replace([np.inf,-np.inf],np.nan)
 z=pd.concat([f,r.iloc[t+1].reindex(f.index)],axis=1).dropna()
 if len(z)>=8:
  rows.append((p.index[t],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
  signals += [{'date':p.index[t],'symbol':s,'signal':float(v)} for s,v in f.dropna().items()]
a=np.array([x[1] for x in rows]); n=np.array([x[2] for x in rows])
print('dates',len(a),'avgN',round(n.mean(),2) if len(n) else 0,'coverage',round(n.mean()/len(u),4) if len(n) else 0)
if len(a)>1:
 print('daily IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
 for lab,cut in [('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
  q=a[[x[0]>=cut for x in rows]]; print(lab,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)) if len(q)>1 else 'insufficient')
 pd.DataFrame(signals).to_csv('scripts/miner_3_20290628_choppy_reversal_signal.csv',index=False); print('signal_artifact scripts/miner_3_20290628_choppy_reversal_signal.csv')

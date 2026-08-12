import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
acct=get_account_dict(); u=acct.get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)==0: d=get_index_daily_data(s,1500)
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); out=[]
for t in range(65,len(p)-1):
 disp=r.iloc[t-19:t+1].std(axis=1).mean(); cross=r.iloc[t-4:t+1].sum(); vol=r.iloc[t-19:t+1].std()
 if not np.isfinite(disp): continue
 f=(-cross/vol).replace([np.inf,-np.inf],np.nan).dropna(); z=pd.concat([f,r.iloc[t+1].reindex(f.index)],axis=1).dropna()
 if len(z)>=8: out.append((p.index[t],z.iloc[:,0].corr(z.iloc[:,1]),len(z),f,disp))
a=np.array([x[1] for x in out]); n=np.array([x[2] for x in out]); print('dispersion-relative reversal dates',len(a),'avgN',n.mean(),'coverage',n.mean()/len(u),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for lab,cut in [('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
 q=a[[x[0]>=cut for x in out]]; print(lab,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
for qtile in [.5,.7]:
 vals=[]
 for i,x in enumerate(out):
  hist=[y[4] for y in out[:i]]
  if len(hist)>30 and x[4]>=np.quantile(hist,qtile): vals.append(x[1])
 vals=np.array(vals); print('conditional q',qtile,'dates',len(vals),'IC %.6f ICIR %.6f'%(vals.mean(),vals.mean()/vals.std(ddof=1)))

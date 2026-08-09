import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 try: d=get_stock_daily_data(s,3000)
 except Exception: d=None
 if d is None:
  try: d=get_index_daily_data(s,3000)
  except Exception: d=None
 if d is not None and len(d): px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); mom=p.pct_change(20)
down=r.where(r<0,0).rolling(20).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)
f=(mom/down).shift(1)
for h in [1,5,10]:
 vals=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],p.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 a=np.array(vals); print('h',h,'dates',len(a),'meanN',len(U),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
vals=[]
for i in range(len(p)-1):
 z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8: vals.append((p.index[i],z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
df=pd.DataFrame(vals,columns=['date','ic','n']); print('coverage %.4f dates %d avgN %.2f'%(df.n.mean()/len(U),len(df),df.n.mean()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=df[(df.date>=a)&(df.date<=b+'-12-31')].ic; print(a,b,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('turnover %.4f'%(f.rank(axis=1,pct=True).diff().abs().mean().mean()))
f.to_csv('scripts/miner_2_20270325_downside_momentum_signal.csv',index_label='date')

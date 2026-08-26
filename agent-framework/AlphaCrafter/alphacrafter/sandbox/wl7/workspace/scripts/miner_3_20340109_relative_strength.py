import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
assets=[os.path.basename(x)[:-4] for x in files]
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
rel=p.pct_change(20).sub(p.pct_change(20).median(axis=1),axis=0)
vol=r.rolling(40).std(); f=rel/vol
for h in [1,5,10,20]:
 vals=[]; dates=[]; ns=[]
 for i in range(len(p)-h):
  t=p.index[i]; nxt=p.index[i+1:i+h+1]
  x=f.loc[t]; y=p.loc[nxt[-1]]/p.loc[nxt[0]]-1
  z=pd.concat([x,y.rename('y')],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.y).statistic); dates.append(t); ns.append(len(z))
 s=pd.Series(vals,index=dates).replace([np.inf,-np.inf],np.nan).dropna()
 print('h',h,'IC %.6f ICIR %.6f hit %.3f dates %d avgN %.2f'%(s.mean(),s.mean()/s.std(),(s>0).mean(),len(s),np.mean(ns)))
print('coverage %.4f'%f.notna().mean().mean())
print('turnover %.4f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('period',p.index.min().date(),p.index.max().date(),'assets',len(assets))

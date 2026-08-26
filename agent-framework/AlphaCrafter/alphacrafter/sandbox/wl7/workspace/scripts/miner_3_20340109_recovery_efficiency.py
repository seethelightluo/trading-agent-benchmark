import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-01-08'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Recovery efficiency: medium-horizon gain divided by downside path risk, with lagged cross-sectional centering
mom=p.pct_change(15)
down=r.clip(upper=0).rolling(30).std()*np.sqrt(30)
f=mom/(down+1e-8)
f=f.sub(f.median(axis=1),axis=0)
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for i in range(len(p)-h):
  x=f.iloc[i]; y=p.iloc[i+h]/p.iloc[i+1]-1
  z=pd.concat([x,y.rename('y')],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.y.nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.y).statistic); ns.append(len(z))
 s=pd.Series(vals).dropna(); print('h',h,'IC %.6f ICIR %.6f hit %.3f dates %d avgN %.2f'%(s.mean(),s.mean()/s.std(),(s>0).mean(),len(s),np.mean(ns)))
print('coverage %.4f turnover %.4f period %s to %s assets %d'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),p.index.min().date(),p.index.max().date(),len(assets)))

import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p):
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px={s:load('../persistent/stock_data/'+s+'.csv') for s in U}
dxy=load('../persistent/index_data/DXY.csv')
R=pd.DataFrame(px).pct_change(); dr=dxy.pct_change()
# negative rolling beta to DXY, using completed t data, predict t+1
F=pd.DataFrame(index=R.index,columns=U,dtype=float)
for s in U:
 F[s]=-(R[s].rolling(60,min_periods=45).cov(dr)/dr.rolling(60,min_periods=45).var())
rows=[]
for i in range(len(R)-1):
 f=F.iloc[i]; y=R.iloc[i+1]
 z=pd.concat([f,y],axis=1).dropna()
 if len(z)>=8: rows.append((R.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
ic=np.array([x[1] for x in rows]); dates=[x[0] for x in rows]
print('dates',len(ic),'avg names',np.mean([pd.concat([F.loc[d],R.loc[d+pd.Timedelta(1,'D')] if False else F.loc[d]],axis=0).notna().sum()/2 for d in []]) if False else '15 universe')
print('daily IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(ic),np.nanmean(ic)/np.nanstd(ic,ddof=1)*np.sqrt(len(ic)) if len(ic)>1 else np.nan,np.mean(ic>0)))
for h in [5,10]:
 vals=[]
 for i in range(len(R)-h):
  y=R.iloc[i+1:i+1+h].sum(); z=pd.concat([F.iloc[i],y],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals); print('%dd IC %.6f ICIR %.6f n %d'%(h,np.mean(a),np.mean(a)/np.std(a,ddof=1)*np.sqrt(len(a)),len(a)))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=ic[(np.array([str(d)[:4] for d in dates])>=a)&(np.array([str(d)[:4] for d in dates])<=b)]
 print(a,b,'n',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1)*np.sqrt(len(q)) if len(q)>1 else np.nan)
# turnover rank
rank=F.rank(axis=1,pct=True); print('coverage',F.notna().mean().mean(),'turnover',np.mean(np.abs(rank.diff()).mean(axis=1).dropna()))

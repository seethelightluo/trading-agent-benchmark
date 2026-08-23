import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
UNIV=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-04-02')
px={}
for s in UNIV:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f); d.date=pd.to_datetime(d.date); d=d[d.date<=cut].set_index('date').sort_index(); px[s]=d.close
p=pd.concat(px,axis=1).sort_index(); r=p.pct_change()
# residual momentum: asset 20d return minus equal-weight universe 20d return, / idio vol; lag one day
m=p.pct_change(20); bench=r.mean(axis=1).rolling(20).sum(); vol=(r.sub(r.mean(axis=1),axis=0)**2).rolling(20).mean().pow(.5)*np.sqrt(20)
f=(m.sub(bench,axis=0)/(vol+1e-8)).shift(1)
rows=[]
for i in range(len(p)-1):
 dt=p.index[i]; valid=f.iloc[i].notna() & p.iloc[i+1].notna()
 if valid.sum()>=8:
  ic=spearmanr(f.iloc[i][valid],r.iloc[i+1][valid]).statistic
  rows.append((dt,ic,valid.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.sum()/(len(x)*len(UNIV)))
print('daily_ic %.8f icir %.8f hit %.4f'%(x.ic.mean(),x.ic.mean()/(x.ic.std(ddof=1)+1e-12), (x.ic>0).mean()))
for h in [5,10,20]:
 rr=p.pct_change(h)
 vals=[]
 for i in range(len(p)-h):
  v=f.iloc[i].notna()&rr.iloc[i+h].notna()
  if v.sum()>=8: vals.append(spearmanr(f.iloc[i][v],rr.iloc[i+h][v]).statistic)
 print('decay',h,np.nanmean(vals),len(vals))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-04-02')]:
 z=x.loc[a:b].ic;print('regime',a,b,len(z),z.mean() if len(z) else np.nan)
# artifact
out=f.copy();out.index.name='date';out.to_csv('scripts/miner_1_20270402_residual_relative_momentum_signal.csv')

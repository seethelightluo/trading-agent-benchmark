import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
assets=sorted(D); px=pd.concat({a:D[a]['close'] for a in assets},axis=1).sort_index(); r=px.pct_change()
# Path-efficiency momentum: directional displacement divided by total traveled path.
# Positive = persistent trend; negative = persistent decline; low magnitude = choppy.
net=r.rolling(10,min_periods=8).sum(); path=r.abs().rolling(10,min_periods=8).sum()
factor=(net/(path+1e-12)).shift(1)
print('assets',len(assets),'range',px.index.min().date(),px.index.max().date())
for h in [1,5,10,20]:
 fr=px.pct_change(h).shift(-h); ics=[]; ns=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(ics); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()))
fr=r.shift(-1); z=[]
for dt in factor.index:
 q=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(q)>=8:z.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
z=pd.DataFrame(z,columns=['date','ic']).set_index('date');print('year',z.groupby(z.index.year).ic.mean().round(5).to_dict())
rank=factor.rank(axis=1,pct=True); print('coverage',factor.notna().sum().sum()/factor.size,'valid_dates',len(z),'valid_avg',factor.notna().sum(axis=1).mean(),'turnover',rank.diff().abs().sum(axis=1).div(len(assets)).dropna().mean())
for w in [60,120,250]:
 a=z.ic.tail(w);print('recent',w,'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()))

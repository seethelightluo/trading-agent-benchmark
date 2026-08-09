import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2026-11-18'); assets=[os.path.basename(x)[:-4] for x in glob.glob('../persistent/stock_data/*.csv')]
F={}; FW={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date'); r=d.close.pct_change()
 # row-wise asset-local path efficiency, avoiding calendar gaps
 F[a]=d.close.pct_change(20)/r.abs().rolling(20,min_periods=16).sum()
 for h in [1,5,10]: FW.setdefault(h,{})[a]=d.close.pct_change(h).shift(-h)
fac=pd.DataFrame(F).sort_index(); fac.to_csv('scripts/miner_3_20261119_path_efficiency_signal.csv')
print('assets',len(assets),'factor rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 fwd=pd.DataFrame(FW[h]).reindex(fac.index); vals=[];ds=[];ns=[]
 for dt in fac.index:
  x=fac.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f cov %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean(),fac.notna().sum(axis=1).mean()/15))
 if h==1: print('regimes',[(y,round(s[s.index.year==y].mean(),5),len(s[s.index.year==y])) for y in range(2020,2027)])
print('turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())

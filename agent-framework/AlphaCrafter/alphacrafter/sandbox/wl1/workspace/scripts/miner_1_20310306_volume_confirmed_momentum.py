import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
assets=[os.path.basename(f)[:-4] for f in files]
px={}; vol={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 px[a]=d.close; vol[a]=d.volume.replace(0,np.nan)
P=pd.DataFrame(px).sort_index(); V=pd.DataFrame(vol).reindex(P.index)
# volume-confirmed risk adjusted momentum, lagged one day
mom=P.pct_change(20); rv=P.pct_change().rolling(20).std()*np.sqrt(252)
vt=np.log(V.rolling(20).mean()/V.rolling(60).mean())
F=(mom/rv)*np.tanh(vt).shift(1)
# use forward horizons, factor at date t predicts t+h return
for h in [1,5,10,20]:
  rows=[]
  for i in range(len(P)-h):
   f=F.iloc[i]; r=P.iloc[i+h]/P.iloc[i]-1
   z=pd.concat([f,r],axis=1).dropna()
   if len(z)>=8: rows.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  x=np.array(rows); print('H',h,'n',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0))
# coverage and turnover via daily cross-sectional ranks
valid=F.notna().sum(axis=1); print('dates',len(P),'avg_names',valid.mean(),'coverage',valid.mean()/len(assets))
r=F.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean())
# years h1
rows=[]
for i in range(len(P)-1):
 z=pd.concat([F.iloc[i],P.iloc[i+1]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[i].year,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['year','ic']); print(df.groupby('year').ic.agg(['mean','count']).round(4).to_string())
# signal artifact
out=pd.DataFrame(F.stack(),columns=['signal']); out.index.names=['date','asset']; out.reset_index().to_csv('scripts/miner_1_20310306_volume_confirmed_momentum_signal.csv',index=False)

import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
P={os.path.basename(f)[:-4]:pd.read_csv(f).set_index('date')['close'] for f in glob.glob('../persistent/stock_data/*.csv')}
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv').set_index('date')['close'].reindex(px.index).ffill()
# Conditional recovery: distance above 40d low, scaled by volatility and activated by elevated VIX z-score.
low=px.rolling(40,min_periods=25).min(); vol=r.rolling(20,min_periods=15).std(); rec=(px/low-1)/(vol*np.sqrt(20)+1e-12)
z=(v-v.rolling(120,min_periods=60).mean())/(v.rolling(120,min_periods=60).std()+1e-12)
for name,f in [('vix_weighted',rec*(1+z.clip(-1,3))),('stress_only',rec.where(z>0)),('recovery_slope', (px/px.shift(5)-1)/(vol*np.sqrt(5)+1e-12) * (1+z.clip(-1,3)))]:
 print('\n',name)
 for h in [1,5,10,20]:
  a=[]; dates=[]; ns=[]
  for i in range(len(px)-h):
   zc=pd.concat([f.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
   if len(zc)>=8 and zc.iloc[:,0].nunique()>1:
    a.append(spearmanr(zc.iloc[:,0],zc.iloc[:,1]).statistic);dates.append(px.index[i]);ns.append(len(zc))
  a=np.array(a); print(h,len(a),round(a.mean(),6),round(a.mean()/a.std(ddof=1),6), 'N',round(np.mean(ns),2),'recent',round(a[-120:].mean(),5),round(a[-120:].mean()/a[-120:].std(ddof=1),4))
 print('coverage',f.notna().mean().mean())

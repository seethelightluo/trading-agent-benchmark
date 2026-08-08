import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
A=sorted(D); C=pd.concat({a:D[a].close for a in A},axis=1).sort_index(); V=pd.concat({a:D[a].volume for a in A},axis=1).sort_index(); R=C.pct_change()
# Persistent abnormal participation: 5-day volume surprise relative to 60-day median, lagged.
f=(V/(V.rolling(60,min_periods=20).median()+1e-12)).rolling(5,min_periods=5).mean().apply(np.log1p).shift(1)
# demean cross section, because only relative participation is intended
f=f.sub(f.mean(axis=1),axis=0)
for h in [1,5,10,20]:
 y=C.pct_change(h).shift(-h); z=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 a=np.array(z);print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()))
y=R.shift(-1); z=[]
for d in f.index:
 q=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(q)>=8:z.append((d,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
z=pd.DataFrame(z,columns=['date','ic']).set_index('date');print('years',z.groupby(z.index.year).ic.mean().round(5).to_dict());print('coverage',round(f.notna().sum().sum()/f.size,4),'avgvalid',round(f.notna().sum(axis=1).mean(),2),'turnover',round(f.rank(axis=1,pct=True).diff().abs().sum(axis=1).div(15).dropna().mean(),5))

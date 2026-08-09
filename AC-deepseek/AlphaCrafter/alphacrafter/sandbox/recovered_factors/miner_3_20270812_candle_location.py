import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
A=sorted(D); O=pd.concat({a:D[a]['open'] for a in A},axis=1).sort_index(); C=pd.concat({a:D[a]['close'] for a in A},axis=1).sort_index(); H=pd.concat({a:D[a]['high'] for a in A},axis=1).sort_index(); L=pd.concat({a:D[a]['low'] for a in A},axis=1).sort_index()
# Close-location reversal: lagged signed candle body normalized by full daily range.
# Strong closes near the high are treated as short-term continuation; inverse tests reversal.
body=(C/O-1); rng=(H-L)/O
f=(body/(rng+1e-8)).shift(1)
print('assets',len(A),'range',C.index.min().date(),C.index.max().date())
for sign in [1,-1]:
 for h in [1,5,10,20]:
  y=C.pct_change(h).shift(-h); z=[];ns=[]
  for d in f.index:
   q=pd.concat([sign*f.loc[d],y.loc[d]],axis=1).dropna()
   if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
  a=np.array(z);print('sign',sign,'H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()))
y=C.pct_change().shift(-1);z=[]
for d in f.index:
 q=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
 if len(q)>=8:z.append((d,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
z=pd.DataFrame(z,columns=['date','ic']).set_index('date');print('years',z.groupby(z.index.year).ic.mean().round(5).to_dict());print('coverage',round(f.notna().sum().sum()/f.size,4),'valid_dates',len(z),'avgvalid',round(f.notna().sum(axis=1).mean(),2),'turnover',round(f.rank(axis=1,pct=True).diff().abs().sum(axis=1).div(15).dropna().mean(),5))
for w in [60,120,250]:
 a=z.ic.tail(w);print('recent',w,'IC %.6f ICIR %.6f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)))

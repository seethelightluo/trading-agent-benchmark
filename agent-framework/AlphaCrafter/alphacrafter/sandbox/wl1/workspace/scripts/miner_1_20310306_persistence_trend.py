import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv'); assets=[os.path.basename(f)[:-4] for f in files]
P=pd.DataFrame({a:pd.read_csv(f,parse_dates=['date']).set_index('date').close for a,f in zip(assets,files)}).sort_index().loc[:'2031-03-06']
r=P.pct_change(); mom=P.pct_change(20); vol=r.rolling(40).std(); persistence=(r.rolling(20).apply(lambda x: np.mean(x>0),raw=True)-.5)*2; F=(mom/vol*persistence).shift(1)
def calc(h):
 vals=[]
 for i in range(len(P)-h):
  z=pd.concat([F.iloc[i],(P.iloc[i+h]/P.iloc[i]-1)],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 x=np.asarray(vals); return len(x),float(np.mean(x)),float(np.mean(x)/np.std(x,ddof=1)),float(np.mean(x>0))
print('range',P.index.min(),P.index.max(),'assets',len(assets));
for h in [1,5,10,20]: print(h,calc(h))
valid=F.notna().sum(axis=1); print('avg names',valid.mean(),'coverage',valid.mean()/len(assets),'dates',len(P)); print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
F.stack().rename('signal').rename_axis(['date','asset']).reset_index().to_csv('scripts/miner_1_20310306_persistence_trend_signal.csv',index=False)

import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
A=sorted(D); X={k:D[k] for k in A}; px=pd.concat({k:X[k].close for k in A},axis=1).sort_index(); r=px.pct_change()
# Candle pressure: signed close location within prior day's range, smoothed over 3 days.
# Uses only completed OHLC and is intended as a short-horizon continuation/reversal test.
loc=pd.concat({k:((X[k].close-X[k].low)/(X[k].high-X[k].low+1e-12)*2-1) for k in A},axis=1).sort_index()
factor=loc.rolling(3,min_periods=2).mean().shift(1)
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h); ic=[];n=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);n.append(len(z))
 a=np.array(ic);print('H',h,'dates',len(a),'avgN',np.mean(n),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()))
y=r.shift(-1); ic=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:ic.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
z=pd.DataFrame(ic,columns=['date','ic']).set_index('date');print('year',z.groupby(z.index.year).ic.mean().round(5).to_dict());print('coverage',factor.notna().sum().sum()/factor.size,'turnover',factor.rank(axis=1,pct=True).diff().abs().sum(axis=1).div(15).mean())
for w in [60,120,250]:
 a=z.ic.tail(w);print('recent',w,a.mean(),a.mean()/a.std(ddof=1))

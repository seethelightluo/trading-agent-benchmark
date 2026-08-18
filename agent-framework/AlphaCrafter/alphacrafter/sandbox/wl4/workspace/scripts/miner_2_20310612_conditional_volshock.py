import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2031-06-11')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close.loc[:cut] for s in U}; dates=sorted(set.intersection(*[set(x.index) for x in D.values()]));P=pd.DataFrame({s:D[s].reindex(dates) for s in U});r=P.pct_change();ratio=r.rolling(5).std()/(r.rolling(20).std()+1e-12); med=r.median(axis=1); disp=r.std(axis=1); F={'down':-ratio*(med<0).astype(float).values[:,None], 'disp':-ratio*(disp>disp.rolling(60).median()).astype(float).values[:,None], 'both':-ratio*((med<0)&(disp>disp.rolling(60).median())).astype(float).values[:,None]}
def go(f,H):
 a=[]
 for i in range(20,len(P)-H-1):
  x=f.iloc[i];y=P.iloc[i+1+H]/P.iloc[i+1]-1;o=x.notna()&y.notna()
  if o.sum()>=8 and x[o].nunique()>1:
   z=spearmanr(x[o],y[o]).statistic
   if np.isfinite(z):a.append(z)
 a=np.array(a);return len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
for n,f in F.items(): print(n,'H10',go(f,10),'H20',go(f,20),'recent',go(f.iloc[-1095:].set_axis(range(1095)),10))

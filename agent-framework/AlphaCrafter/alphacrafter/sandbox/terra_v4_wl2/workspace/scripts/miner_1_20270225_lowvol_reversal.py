import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.sort_index() for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); px=pd.DataFrame(p)
# Low-volatility carry: negative recent return, scaled by stable 20d volatility; point-in-time.
vol=r.rolling(20,min_periods=15).std(); f=-(px.pct_change(5))/vol
# cross-sectional median neutralization
f=f.sub(f.median(axis=1),axis=0)
rows=[]
for h in [1,5,10]:
 y=px.pct_change(h).shift(-h)
 for d in f.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): rows.append((d,h,q,len(z)))
o=pd.DataFrame(rows,columns=['date','h','ic','n']);print('assets',len(A),'period',o.date.min(),o.date.max())
for h in [1,5,10]:
 x=o[o.h==h]; print('H',h,'dates',len(x),'avgN',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC %.8f ICIR %.8f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean()))
 for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2027-12-31')]:
  z=x[(x.date>=lo)&(x.date<=hi)]; print('REG',lo,len(z),round(z.ic.mean(),6),round(z.ic.mean()/z.ic.std(ddof=1),6) if len(z)>1 else np.nan)
# artifact, full factor values
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('../persistent/factor_signals_miner_1_20270225_lowvol_reversal.csv')
print('artifact written')

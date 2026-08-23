import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-05-10')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date').close.astype(float) for s in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); r5=p.pct_change(5); v20=r.rolling(20).std(); v60=r.rolling(60).std(); shock=v20/(v60+1e-12)
# short-horizon reversal, stronger after an idiosyncratic volatility shock; lagged to avoid lookahead
sig=(-r5*shock).shift(1)
for h in [5,10,20,40]:
 f=p.shift(-h)/p-1; rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(o),'avg_names',round(o.n.mean(),3),'coverage',round(o.n.sum()/(len(o)*15),6),'IC %.8f ICIR %.8f hit %.6f'%(o.ic.mean(),o.ic.mean()/o.ic.std(ddof=1),np.mean(o.ic>0)))
 for a,b in [('2020','2024'),('2025','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
  q=o.loc[a:b]
  if len(q): print('regime',a,b,len(q),round(q.ic.mean(),8),round(q.ic.mean()/q.ic.std(ddof=1),6),round((q.ic>0).mean(),4))
 if h==10:o.to_csv('scripts/miner_1_20340511_volshock_reversal_5d_signal.csv')
print('turnover',round(float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))

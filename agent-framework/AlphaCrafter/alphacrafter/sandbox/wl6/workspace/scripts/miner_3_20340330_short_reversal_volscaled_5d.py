import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-03-29')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date').close.astype(float) for s in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); vol=r.rolling(20).std();
# Contrarian shock: recent 5d loss receives higher score, normalized by medium-term volatility.
sig=-p.pct_change(5)/(vol*np.sqrt(5)+1e-12)
rows=[]
for h in [5,10,20,40]:
 fwd=p.shift(-h)/p-1; vals=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 out=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(out),'avg_names',round(out.n.mean(),3),'coverage',round(out.n.sum()/(len(out)*15),6),'IC %.8f ICIR %.8f hit %.6f'%(out.ic.mean(),out.ic.mean()/out.ic.std(ddof=1),np.mean(out.ic>0)))
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2034')]:
  q=out.loc[a:b]; print('regime',a,b,len(q),round(q.ic.mean(),8),round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else np.nan,round((q.ic>0).mean(),4))
 if h==10: out.to_csv('scripts/miner_3_20340330_short_reversal_volscaled_5d_signal.csv')
print('turnover',round(float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))

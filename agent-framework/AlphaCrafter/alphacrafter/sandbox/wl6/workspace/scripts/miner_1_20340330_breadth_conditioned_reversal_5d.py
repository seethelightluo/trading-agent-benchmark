import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2034-03-29')
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').drop_duplicates('date').sort_values('date').set_index('date').close.astype(float) for s in U}
p=pd.DataFrame(px).sort_index(); r=p.pct_change();
# Contrarian short-horizon reversal, volatility scaled, with a smooth market-breadth regime weight.
r5=p.pct_change(5); v20=r.rolling(20).std()*np.sqrt(252)+1e-12
bread=r.gt(0).rolling(20).mean().mean(axis=1)
# Reversal is emphasized in weak breadth and mildly suppressed in strongly trending breadth.
gate=(1.25-(bread-0.5).abs()).clip(0.75,1.25)
sig=(-r5/v20).mul(gate,axis=0).shift(1)

def calc(h):
 fwd=p.shift(-h)/p-1; rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
out=calc(10); a=out.ic
print('dates',len(out),'avg_names',round(out.n.mean(),3),'coverage',round(out.n.mean()/15,6))
print('IC %.8f ICIR %.8f hit %.6f turnover %.6f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [5,10,20,40]:
 q=calc(h); print('decay',h,'IC %.8f ICIR %.8f dates %d'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),len(q)))
for aa,bb in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2029'),('2030','2034')]:
 q=out.loc[aa:bb]; print('regime',aa,bb,'dates',len(q),'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean()))
sig.tail(1).T.to_csv('scripts/miner_1_20340330_breadth_conditioned_reversal_5d_signal.csv')

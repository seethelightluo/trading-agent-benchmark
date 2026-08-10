import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}
P=pd.concat(D,axis=1).sort_index().loc[:cut]; R=P.pct_change()
# rate-spread regime: use only lagged rates, amplitude clipped; asset reversal is conditional on rates divergence
spread=R['US10Y'].rolling(20,min_periods=10).sum()-R['CN10Y'].rolling(20,min_periods=10).sum()
reg=np.tanh(spread/0.08).shift(1)
F=-(P.shift(1)/P.shift(4)-1).mul(reg,axis=0)
# residualize common cross-sectional component to avoid market direction
F=F.sub(F.median(axis=1),axis=0)
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; out=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: out.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); q=a.ic
 print('H',h,'dates',len(q),'N',a.n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 if h==1:
  for k,g in q.groupby(q.index.year): print('YR',k,len(g),g.mean(),g.mean()/g.std(ddof=1))
print('coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
F.stack().rename('signal').rename_axis(['date','symbol']).to_csv('scripts/miner_3_20261217_rate_spread_reversal_signal.csv')

import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=5000).set_index('date')['close'].astype(float) for s in syms}
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Short reversal is activated only when cross-asset dispersion is elevated;
# normalize by each asset's recent downside risk and lag all inputs.
csdisp=r.rolling(20).std().mean(axis=1).shift(1)
base=-(P.pct_change(5).shift(1))/(r.rolling(20).std().shift(1)+1e-8)
threshold=csdisp.rolling(252,min_periods=120).median()
F=base.where(csdisp>=threshold,0.0)
F=F.replace([np.inf,-np.inf],np.nan)
F.to_csv('scripts/miner_1_20350830_dispersion_gated_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.shift(-h)/P-1; cs=[];ns=[];dates=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):cs.append(c);ns.append(len(z));dates.append(d)
 a=pd.Series(cs); print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(dates).date()} end={max(dates).date()} turnover={F.rank(axis=1,pct=True).diff().abs().mean().mean():.5f}')

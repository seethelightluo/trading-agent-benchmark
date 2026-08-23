import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; prices={}
for s in U:
 p=os.path.join(base,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p); dc={c.lower():c for c in d.columns}; prices[s]=pd.Series(pd.to_numeric(d[dc['close']],errors='coerce').values,index=pd.to_datetime(d[dc['date']]))
px=pd.DataFrame(prices).sort_index().ffill(); r=np.log(px/px.shift(1)); ret20=r.rolling(20).sum().shift(1); down=r.where(r<0,0).pow(2).rolling(20).mean().shift(1).pow(.5); up=r.where(r>0,0).pow(2).rolling(20).mean().shift(1).pow(.5); asym=(down/(up+1e-8)).clip(0,10); vol=r.rolling(40).std().shift(1); f=(-(ret20)/(vol+1e-8))*(1+0.35*(asym-1).clip(-1,2))
def calc(h):
 rr=r.rolling(h).sum().shift(-h); xs=[]
 for dt in px.index:
  z=pd.concat([f.loc[dt].rename('f'),rr.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: xs.append(spearmanr(z.f,z.y).statistic)
 return pd.Series(xs,dtype=float)
for h in [5,10,20,40]:
 x=calc(h); print('h%d dates=%d IC=%.8f ICIR=%.8f hit=%.4f'%(h,len(x),x.mean(),x.mean()/(x.std(ddof=1)+1e-12),(x>0).mean()))
print('universe=%d coverage=%.4f turnover=%.6f'%(len(prices),px.notna().mean().mean(),(f.rank(axis=1,pct=True).diff().abs().mean(axis=1)).mean()))
f.index.name='date'; f.to_csv('scripts/miner_2_20350830_downside_asymmetry_reversal_signal.csv')

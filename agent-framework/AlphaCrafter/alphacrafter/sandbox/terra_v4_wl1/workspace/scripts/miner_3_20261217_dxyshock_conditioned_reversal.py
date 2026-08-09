import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; ib='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().ffill(); P=P[P.index<=cut]
dxy=pd.read_csv(f'{ib}/DXY.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
R=P.pct_change(); r3=R.rolling(3,min_periods=3).sum(); shock=dxy.pct_change(5); hist=shock.shift(1).rolling(120,min_periods=60); z=(shock-hist.mean())/hist.std(); gate=z>0.75
F=-r3.where(gate,0.0); F=F.sub(F.median(axis=1),axis=0)
for label, sl in [('full',slice(None)),('recent',slice(pd.Timestamp('2024-01-01'),None)),('post',slice(pd.Timestamp('2026-07-16'),None))]:
 f=F.loc[sl]; fr=R.shift(-1).loc[f.index]; vals=[]; ns=[]
 for d in f.index:
  x=f.loc[d]; y=fr.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
 a=np.array(vals); print(label,'dates',len(a),'avg_n',round(float(np.mean(ns)),2),'IC',round(float(np.nanmean(a)),6),'ICIR',round(float(np.nanmean(a)/np.nanstd(a,ddof=1)),6),'hit',round(float(np.mean(a>0)),4))
print('coverage',round(float(F.notna().mean().mean()),4),'turnover',round(float(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
F.to_csv('scripts/miner_3_20261217_dxyshock_conditioned_reversal_signal.csv',index_label='date')

import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
base='../persistent/stock_data'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 f=f'{base}/{s}.csv'
 d=pd.read_csv(f,parse_dates=['date']).set_index('date')
 px[s]=d['close'].pct_change()
r=pd.DataFrame(px).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
# shock reversal: contrarian 3d return, normalized by 20d vol, activated by positive VIX 5d change
rv=r.rolling(20).std()
raw=-(r.rolling(3).sum())/(rv*np.sqrt(3)+1e-8)
vixshock=(vix.pct_change(5)>0).astype(float)
sig=raw.mul(vixshock.reindex(r.index).fillna(0),axis=0)
# forward returns; cross-sectional IC at each date
outs=[]
for h in [1,5,10]:
 fwd=r.shift(-h).rolling(h).sum().shift(-(h-1))
 vals=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(a)>=8: vals.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 x=pd.Series(vals).dropna(); print('h',h,'dates',len(x),'Nmean',np.nanmean([len(pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()) for d in sig.index if len(pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna())>=8]),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
 if h==1:
  print('regimes',[(p, x.loc[[d for d in x.index if str(d)[:4] in ys]].mean(),len(x.loc[[d for d in x.index if str(d)[:4] in ys]])) for p,ys in [('20-22',['2020','2021','2022']),('23-24',['2023','2024']),('25-27',['2025','2026','2027'])]])
# artifact
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_1_20270325_vix_shock_reversal_signal.csv',index=False)
print('coverage',sig.notna().mean().mean(),'active', (sig!=0).mean().mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())

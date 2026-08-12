import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').close for a in assets}).sort_index()
p=p.loc[:'2031-08-07']; r20=p.pct_change(20); f=r20.sub(r20.mean(axis=1),axis=0).shift(1); fr=p.shift(-5).div(p).sub(1)
ics=[];dates=[];turn=[]; cov=[]; prev=None
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic;ics.append(q);dates.append(dt);cov.append(len(z)/15)
  rr=f.loc[dt].rank(pct=True)
  if prev is not None:turn.append((rr-prev).abs().mean())
  prev=rr
ics=np.array(ics); print('idea=market_relative_momentum_20d_h5','dates',len(ics),'avg_n',np.mean(np.array(cov)*15),'coverage',np.mean(cov),'IC',np.mean(ics),'ICIR',np.mean(ics)/np.std(ics,ddof=1),'hit',np.mean(ics>0),'turnover',np.mean(turn))
for lab,lo,hi in [('2020-22','2020','2022'),('2023-25','2023','2025'),('2026-31','2026','2031')]:
 a=ics[[str(d)[:4]>=lo and str(d)[:4]<=hi for d in dates]];print(lab,len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1))
out='scripts/miner_1_20310807_residual_momentum_signal.csv';f.reset_index().to_csv(out,index=False);print('artifact',out)

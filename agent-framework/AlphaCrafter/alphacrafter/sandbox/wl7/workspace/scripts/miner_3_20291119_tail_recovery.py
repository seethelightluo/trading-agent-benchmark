import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-11-18'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=P.pct_change()
# Tail-asymmetric recovery: prior 20d return divided by downside deviation, then penalize current drawdown from 60d peak.
down=R.where(R<0).rolling(30,min_periods=12).std().shift(1)
peak=P.rolling(60,min_periods=30).max().shift(1)
sig=((P.shift(1)/P.shift(21)-1)/(down+1e-8))*(1-(peak-P.shift(1))/peak)
sig=sig.replace([np.inf,-np.inf],np.nan)
print('rows',len(P),'assets_min',P.notna().sum().min())
def calc(h, mask=None):
 f=P.shift(-h)/P-1; out=[]; ns=[]
 idx=P.index if mask is None else P.index[mask]
 for dt in idx:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1: out.append(spearmanr(z.x,z.y).statistic); ns.append(len(z))
 q=pd.Series(out).dropna(); return len(q),np.mean(ns),np.mean(q),np.mean(q)/q.std(ddof=1),(q>0).mean()
for h in [5,10,20,40]:
 n,av,ic,ir,hit=calc(h); print('H',h,'dates',n,'avg_n',round(av,2),'coverage',round(av/15,4),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4))
for lab,mask in [('early',P.index<='2026-12-31'),('mid',(P.index>='2027-01-01')&(P.index<='2028-12-31')),('late',P.index>='2029-01-01')]:
 n,av,ic,ir,hit=calc(20,mask); print(lab,'dates',n,'avg_n',round(av,2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(hit,4))
rank=sig.rank(axis=1,pct=True); t=[]
for a,b in zip(rank.index[:-1],rank.index[1:]):
 z=pd.concat([rank.loc[a],rank.loc[b]],axis=1).dropna()
 if len(z): t.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('turnover',round(float(np.mean(t)),6),'valid_dates',sig.dropna(how='all').shape[0])
sig.to_csv('scripts/miner_3_20291119_tail_recovery_signal.csv')

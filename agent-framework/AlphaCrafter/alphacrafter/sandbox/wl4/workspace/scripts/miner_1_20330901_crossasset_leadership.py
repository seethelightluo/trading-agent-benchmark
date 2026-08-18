import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'
p={a:pd.read_csv(f'{B}/{a}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in A}; c=pd.DataFrame(p).sort_index().ffill(); r=c.pct_change()
# persistent leadership: medium return relative to universe, scaled by idiosyncratic residual vol
m=r.rolling(30,min_periods=20).sum(); cross=m.sub(m.mean(axis=1),axis=0); vol=r.rolling(30,min_periods=20).std(); s=(cross/(vol+1e-8)).shift(1); y=c.shift(-10)/c-1
z=[]; ns=[]
for d in s.index:
 ok=s.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8:z.append(spearmanr(s.loc[d][ok],y.loc[d][ok]).statistic);ns.append(ok.sum())
z=np.array(z);print(f'dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1)*np.sqrt(252):.6f} hit={np.mean(z>0):.4f} coverage={np.mean(ns)/15:.4f}')
for n in [260,520,780]:
 q=z[-n:];print(f'recent{n}: IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.6f} N={len(q)}')

import pandas as pd, numpy as np, glob, json
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}).sort_index(); r=p.pct_change(); ur=(1+r.mean(axis=1)).rolling(20).apply(np.prod,raw=True)-1
cand=p.pct_change(20).sub(ur,axis=0)
for f in glob.glob('factors/*.json'):
 if '.bak' in f: continue
 d=json.load(open(f)); fid=d['factor_id']; expr=d['calculation']['expression']
 if 'volume' in expr: continue
 if 'rolling_std(close.pct_change(), 5)' in expr: sig=-p.pct_change(5)/r.rolling(5).std()
 elif 'rolling_std(close.pct_change(), 20)' in expr: sig=p.pct_change(20)/r.rolling(20).std()
 else: continue
 x=cand.stack(); y=sig.stack(); m=x.notna()&y.notna(); print(fid,spearmanr(x[m],y[m]).statistic,m.sum())
print('candidate',cand.stack().shape)

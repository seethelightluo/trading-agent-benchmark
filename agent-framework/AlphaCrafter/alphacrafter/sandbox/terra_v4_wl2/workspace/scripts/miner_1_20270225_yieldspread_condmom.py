import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; root='../persistent'
def load(s): return pd.read_csv(f'{root}/stock_data/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:'2027-02-25']
P=pd.concat({s:load(s) for s in U},axis=1).sort_index(); R=P.pct_change(); sp=(P.US10Y-P.CN10Y).ffill(); sr=sp.diff(20)
cs=R.rolling(20,min_periods=15).sum(); vol=R.rolling(20,min_periods=15).std(); raw=cs.div(vol.replace(0,np.nan)); state=np.sign(sr.shift(1)).replace(0,np.nan); f=raw.mul(state,axis=0)
rows=[]
for h in [1,5,10]:
 fw=P.shift(-h)/P-1
 for d in P.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: rows.append((d,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 q=pd.DataFrame(rows,columns=['date','h','n','ic']); q=q[q.h==h]; ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
 print('h',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((q.ic>0).mean(),4))
out=f.stack().rename('signal').reset_index(); out.columns=['date','asset','signal']; out.to_csv('../persistent/factor_signals_miner_1_20270225_yieldspread_condmom.csv',index=False)
print('artifact',len(out),'coverage',round(out.signal.notna().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))

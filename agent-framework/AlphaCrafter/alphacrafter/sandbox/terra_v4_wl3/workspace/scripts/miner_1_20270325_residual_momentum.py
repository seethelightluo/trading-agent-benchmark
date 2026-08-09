import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); d[s]=x
prices=pd.DataFrame({s:x.close for s,x in d.items()}); rets=prices.pct_change()
# residual, volatility-normalized medium-term momentum; all inputs through t, predicts t+1
mom=prices.pct_change(20); vol=rets.rolling(20).std()*np.sqrt(252)
fac=(mom.sub(mom.median(axis=1),axis=0)).div(vol.replace(0,np.nan))
rows=[]
for t in fac.index:
 f=fac.loc[t]; y=rets.shift(-1).loc[t]
 ok=f.notna()&y.notna()
 if ok.sum()>=8:
  rows.append([t,ok.sum(),spearmanr(f[ok],y[ok]).statistic]+[spearmanr(f[ok],rets.shift(-h).loc[t][ok]).statistic if (rets.shift(-h).loc[t][ok].notna().sum()>=8) else np.nan for h in [5,10]])
r=pd.DataFrame(rows,columns=['date','n','ic1','ic5','ic10']).set_index('date')
for c in ['ic1','ic5','ic10']:
 z=r[c].dropna(); print(c,'dates',len(z),'avgN',r.loc[z.index,'n'].mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
print('coverage',fac.notna().sum(axis=1).mean()/15,'turnover', (fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 z=r.loc[a:b,'ic1'].dropna(); print(a,b,len(z),z.mean(),z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
# artifact
out=fac.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20270325_residual_momentum_signal.csv',index=False)

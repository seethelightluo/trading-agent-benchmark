import pandas as pd, numpy as np, os, json
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index(); r=np.log(P).diff(); m=r.mean(axis=1)
beta=r.rolling(90,min_periods=60).cov(m).div(m.rolling(90,min_periods=60).var(),axis=0).shift(1)
res=r-beta.mul(m,axis=0); rv=res.rolling(60,min_periods=40).std().shift(1)
sig=-(res.rolling(30,min_periods=20).sum().shift(1))/(rv*np.sqrt(30)+1e-12)
h=20; fwd=np.log(P.shift(-h)/P); rows=[]
for d in sig.index:
 z=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=x.ic.mean(); icir=ic/x.ic.std()
# signal artifact contains only information available at decision date
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('../persistent/miner_1_20350105_slow_beta_residual_reversal_signal.csv',index=False)
print('dates',len(x),'mean_n',x.n.mean(),'IC',ic,'ICIR',icir,'hit',(x.ic>0).mean(),'coverage',out.shape[0]/(len(sig)*len(U)),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for label, sl in [('2020-2024',x.loc['2020':'2024']),('2025-2029',x.loc['2025':'2029']),('2030-2032',x.loc['2030':'2032']),('2033-2035',x.loc['2033':'2035'])]:
 print(label,'dates',len(sl),'IC',sl.ic.mean() if len(sl) else np.nan,'ICIR',sl.ic.mean()/sl.ic.std() if len(sl)>1 else np.nan)
print('artifact','../persistent/miner_1_20350105_slow_beta_residual_reversal_signal.csv')

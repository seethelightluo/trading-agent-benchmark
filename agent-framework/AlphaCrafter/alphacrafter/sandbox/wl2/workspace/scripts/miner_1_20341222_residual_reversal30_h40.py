import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index(); R=P.pct_change(); h=40
m=P.pct_change(30); v=R.rolling(90).std()*np.sqrt(252); f=-(m.sub(m.median(axis=1),axis=0)).div(v)
rows=[]
for dt in f.index:
 j=P.index.searchsorted(dt,side='right')
 if j+h-1>=len(P): continue
 z=pd.concat([f.loc[dt],P.iloc[j+h-1]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(x),'mean_n',x.n.mean(),'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(),'hit',(x.ic>0).mean())
for a,b in [('2020','2024'),('2025','2029'),('2030','2032'),('2033','2034')]:
 y=x.loc[a:b]; print(a,len(y),y.ic.mean(),y.ic.mean()/y.ic.std() if len(y)>1 else np.nan,(y.ic>0).mean())
print('turnover',f.rank(axis=1).diff().abs().mean(axis=1).mean()/len(U))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('../persistent/miner_1_20341222_residual_reversal30_h40_signal.csv',index=False)

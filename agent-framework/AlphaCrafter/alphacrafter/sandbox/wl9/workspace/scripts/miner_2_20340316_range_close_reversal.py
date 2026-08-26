import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
close={}; clv={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d['date']=pd.to_datetime(d.date); d=d.set_index('date'); close[s]=d.close.astype(float); clv[s]=((d.close-d.low)/(d.high-d.low).replace(0,np.nan)-.5).rolling(10).mean()
p=pd.DataFrame(close).sort_index(); f=pd.DataFrame(clv).reindex(p.index).shift(1); fw={h:p.shift(-h)/p-1 for h in [10,20,40,60]}
for h,y in fw.items():
 vals=[];ns=[]
 for dt in f.index:
  a=f.loc[dt];b=y.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8: vals.append(spearmanr(-a[ok],b[ok]).statistic);ns.append(ok.sum())
 x=pd.Series(vals).dropna(); print(h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(x.mean(),x.mean()/x.std(),(x>0).mean(),len(x),np.mean(ns)))
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.signal=-out.signal;out.to_csv('scripts/miner_2_20340316_range_close_reversal_signal.csv',index=False)

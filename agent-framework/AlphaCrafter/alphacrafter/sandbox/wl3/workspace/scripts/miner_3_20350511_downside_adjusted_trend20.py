import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'
p={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv')); d['date']=pd.to_datetime(d.date); p[s]=d.set_index('date').close.astype(float)
c=pd.DataFrame(p).sort_index(); c=c.loc[c.index<=pd.Timestamp('2035-05-11')]; r=c.pct_change()
# Reward persistent positive trend while penalizing downside volatility.
down=r.where(r<0,0).rolling(30,min_periods=20).std(); trend=c.pct_change(20); s=trend/(down*np.sqrt(20)+1e-12)
fwd={h:c.pct_change(h).shift(-h) for h in [1,5,10,20]}
rows=[]
for dt in s.index:
 ok=s.loc[dt].notna()&fwd[5].loc[dt].notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(s.loc[dt][ok],fwd[5].loc[dt][ok]).statistic,ok.sum()))
r0=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=r0.ic.dropna()
print('factor=downside_adjusted_trend20');print('dates',len(r0),'instruments',15,'avg_n',r0.n.mean(),'coverage',r0.n.mean()/15)
print('daily_ic %.8f daily_icir %.8f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),(a>0).mean()))
for h in [1,5,10,20]:
 z=[]
 for dt in s.index:
  ok=s.loc[dt].notna()&fwd[h].loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(s.loc[dt][ok],fwd[h].loc[dt][ok]).statistic)
 print('horizon',h,'ic',np.nanmean(z),'n',len(z))
for name,x in [('early',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:]),('recent120',a.tail(120))]: print(name,len(x),'ic',x.mean(),'icir',x.mean()/(x.std(ddof=1)+1e-12)*np.sqrt(len(x)),'hit',(x>0).mean())
print('turnover',s.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'period',r0.index.min().date(),r0.index.max().date())
# Signal artifact for deterministic audit
sig=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();sig.to_csv('scripts/miner_3_20350511_downside_adjusted_trend20_signal.csv',index=False)

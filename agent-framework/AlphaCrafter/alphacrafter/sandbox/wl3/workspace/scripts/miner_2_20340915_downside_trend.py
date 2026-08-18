import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
DATA='../persistent/stock_data'; END=pd.Timestamp('2034-09-15')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in syms:
 p=os.path.join(DATA,s+'.csv')
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).set_index('date'); prices[s]=d.loc[d.index<=END,'close'].astype(float)
px=pd.DataFrame(prices).sort_index(); r=px.pct_change()
trend=px.pct_change(40); down=r.where(r<0).rolling(30,min_periods=15).std(); confirm=px.pct_change(10)
f=-trend/(down*np.sqrt(30)+1e-8) * (1+0.25*np.tanh(confirm*8))
rows=[]
for i in range(len(px)-10):
 vals=f.iloc[i]; fw=px.iloc[i+10]/px.iloc[i]-1; ok=vals.notna()&fw.notna()
 if ok.sum()>=8: rows.append((px.index[i],spearmanr(vals[ok],fw[ok]).statistic,ok.sum()))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for name,z in [('full',x),('recent756',x.tail(756)),('recent252',x.tail(252)),('recent120',x.tail(120))]: print(name,'dates',len(z),'nmean',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
rank=f.rank(axis=1,pct=True); ch=(rank.diff(10).abs().sum(axis=1)/rank.notna().sum(axis=1)).dropna()
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(ch.mean(),4),'last',x.index[-1].date(),'assets',len(px.columns))
f.to_csv('../persistent/miner_2_20340915_downside_trend_signal.csv'); x.to_csv('../persistent/miner_2_20340915_downside_trend_ic.csv')

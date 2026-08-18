import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
trad=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in trad:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 px[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float)
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Relative 40-session trend, volatility scaled; breadth is trailing fraction of assets with positive 20d return.
rel=(p/p.shift(40)-1).sub((p/p.shift(40)-1).median(axis=1),axis=0)
vol=r.rolling(60,min_periods=35).std()*np.sqrt(60)
breadth=(p.pct_change(20)>0).mean(axis=1)
# Smooth, bounded breadth gate rewards trend when participation is broad and damps it when narrow.
gate=(0.5+0.8*(breadth-0.5)).clip(0.2,0.8)
f=(rel/vol).mul(gate,axis=0).shift(1)
rows=[]
for i in range(len(p)-10):
 dt=p.index[i]
 if dt<pd.Timestamp('2026-07-20'): continue
 x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,int(ok.sum())))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(z)*15))
print('IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for w in [120,260,520,780]:
 q=z.tail(w); print('window',w,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
rank=f.rank(axis=1,pct=True); print('rank_turnover',rank.diff().abs().mean(axis=1).dropna().mean())
out='scripts/artifacts'; os.makedirs(out,exist_ok=True)
f.loc[z.index].reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv(out+'/miner_2_20340525_breadth_gated_relative_trend_signal.csv',index=False)
z.reset_index().to_csv(out+'/miner_2_20340525_breadth_gated_relative_trend_ic.csv',index=False)

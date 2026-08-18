import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; q={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'; f=f if os.path.exists(f) else '../persistent/index_data/'+s+'.csv'
 q[s]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].astype(float)
p=pd.DataFrame(q).sort_index(); r=p.pct_change(); ret=p.pct_change(40); rel=ret.sub(ret.median(axis=1),axis=0); vol=r.rolling(60,min_periods=35).std()*np.sqrt(60); breadth=(p.pct_change(20)>0).mean(axis=1)
# Reversal is stronger in narrow participation, but remains continuous and bounded.
gate=(1.3-1.6*(breadth-.5)).clip(.2,1.8); sig=(-rel/vol).mul(gate,axis=0).shift(1)
rows=[]
for i in range(len(p)-10):
 if p.index[i]<pd.Timestamp('2026-07-20'): continue
 x=sig.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((p.index[i],spearmanr(x[ok],y[ok]).statistic,int(ok.sum())))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(z)*15)); print('IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean());
for w in [120,260,520,780]:
 a=z.tail(w); print('window',w,'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
print('rank_turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()); os.makedirs('scripts/artifacts',exist_ok=True); sig.loc[z.index].reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/artifacts/miner_2_20340525_breadth_gated_relative_reversal_signal.csv',index=False); z.reset_index().to_csv('scripts/artifacts/miner_2_20340525_breadth_gated_relative_reversal_ic.csv',index=False)

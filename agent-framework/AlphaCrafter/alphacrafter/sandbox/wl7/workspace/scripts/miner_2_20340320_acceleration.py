import pandas as pd, numpy as np, os, glob
from scipy.stats import spearmanr
cut=pd.Timestamp('2034-03-19')
files=glob.glob('../persistent/stock_data/*.csv')
assets=[os.path.basename(x)[:-4] for x in files]
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index().loc[:cut]
r=px.pct_change()
# Volatility-normalized return acceleration: recent 5d return versus the
# average daily pace of the preceding 20d trend, scaled by 30d risk.
trend20=r.rolling(20).sum()/4
acc=(r.rolling(5).sum()-trend20)/(r.rolling(30).std()*np.sqrt(5)+1e-8)
f=acc.shift(1)
print('assets',len(assets),'dates',len(px),'period',px.index.min().date(),px.index.max().date())
for h in [1,5,10,20]:
    vals=[]; ns=[]
    for i in range(len(px)-h-1):
        z=pd.concat([f.iloc[i],(px.iloc[i+h+1]/px.iloc[i+1]-1).rename('y')],axis=1).dropna()
        if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.y.nunique()>1:
            vals.append(spearmanr(z.iloc[:,0],z.y).statistic); ns.append(len(z))
    s=pd.Series(vals)
    print('h',h,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),3),'dates',len(s),'avgN',round(np.mean(ns),2))
rank=f.rank(axis=1,pct=True)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').dropna().to_csv('scripts/miner_2_20340320_acceleration_signal.csv',index=False)

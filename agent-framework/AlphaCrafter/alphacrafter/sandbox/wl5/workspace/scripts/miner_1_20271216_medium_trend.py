import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F=[]; Y=[]
for s in syms:
    x=pd.read_csv('../persistent/stock_data/'+s+'.csv',usecols=['date','close'])
    x.date=pd.to_datetime(x.date); x=x.set_index('date').sort_index()
    ret=x.close.pct_change(60)
    vol=x.close.pct_change().rolling(40,min_periods=25).std()*np.sqrt(252)
    # medium-horizon trend strength, with volatility normalization and mild recent shock dampening
    sig=(ret/vol.clip(lower=.002)).replace([np.inf,-np.inf],np.nan)
    F.append(sig.rename(s)); Y.append((x.close.shift(-10)/x.close-1).rename(s))
fp=pd.concat(F,axis=1); yp=pd.concat(Y,axis=1); rows=[]
for dt in fp.index:
    z=pd.concat([fp.loc[dt],yp.loc[dt]],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
        rows.append((dt,z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'avg_names',round(a.n.mean(),2),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4),'coverage',round(a.n.sum()/(len(a)*15),4))
print('regimes',[(f'{y}-{y+1}',round(a.ic[(a.index.year>=y)&(a.index.year<=y+1)].mean(),4),int(((a.index.year>=y)&(a.index.year<=y+1)).sum())) for y in [2020,2022,2024,2026]])
print('decay10',round(a.ic.mean(),6))
fp.stack().rename('signal').reset_index().rename(columns={'level_1':'asset'}).to_csv('scripts/miner_1_20271216_medium_trend_signal.csv',index=False)

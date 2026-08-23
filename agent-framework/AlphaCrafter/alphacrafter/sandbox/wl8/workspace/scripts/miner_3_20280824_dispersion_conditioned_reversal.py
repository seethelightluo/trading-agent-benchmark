import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-08-22')
P={}
for s in U:
    x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date'])
    P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change()
vol=r.rolling(20,min_periods=15).std().shift(1)
base=-(r.rolling(5,min_periods=5).sum().shift(1)/vol)
disp=r.std(axis=1).where(r.count(axis=1)>=8).shift(1)
threshold=disp.rolling(60,min_periods=30).median().shift(1)
active=(disp>threshold).astype(float)
sig=base.mul(active,axis=0)
for h in [1,3,5,10]:
    f=px.shift(-h)/px-1; rows=[]
    for d in px.index:
        g=pd.DataFrame({'s':sig.loc[d],'f':f.loc[d]}).dropna()
        if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:
            rows.append((d,spearmanr(g.s,g.f).statistic,len(g),int(active.loc[d])))
    z=pd.DataFrame(rows,columns=['date','ic','n','active']).set_index('date')
    q=z[z.index>=END-pd.Timedelta(days=180)]
    print('h',h,'dates',len(z),'avgN',round(z.n.mean(),2),'active',round(z.active.mean(),3),'IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1),(z.ic>0).mean()),'recentIC %.6f recentICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('artifact_dates',len(sig.index),'panel_coverage',round(sig.notna().sum().sum()/sig.size,4),'nonzero_fraction',round((sig!=0).sum().sum()/sig.size,4))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20280824_dispersion_conditioned_reversal_signal.csv',index=False)

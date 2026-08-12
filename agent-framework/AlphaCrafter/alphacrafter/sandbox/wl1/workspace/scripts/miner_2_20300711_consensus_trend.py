import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    try: d=get_stock_daily_data(s,4000)
    except Exception: d=None
    if d is None or len(d)<200:
        try: d=get_index_daily_data(s,4000)
        except Exception: d=None
    if d is None:return None
    return d[['date','close']].drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame({s:load(s) for s in syms}).sort_index().loc['2020-01-01':]
r=np.log(px).diff(); ret20=px.pct_change(20); ret60=px.pct_change(60); ret120=px.pct_change(120)
vol=r.rolling(40,min_periods=25).std()*np.sqrt(40); down=r.where(r<0,0).rolling(60,min_periods=35).std()*np.sqrt(60)
agree=np.sign(ret20)+np.sign(ret60)+np.sign(ret120)
f=(0.45*ret20/(vol+0.01)+0.35*ret60/(vol+0.01)+0.20*ret120/(vol+0.01))*(1+0.25*np.maximum(agree.abs()-1,0))/(1+2*down)
f=f.replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,5,10,20]:
    fw=px.shift(-h)/px-1; rows=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
        if len(z)>=8: rows.append((dt,z.iloc[:,0].rank().corr(z.iloc[:,1].rank()),len(z)))
    q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
    print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={q.ic.mean():.6f} ICIR={q.ic.mean()/q.ic.std(ddof=1):.6f} hit={(q.ic>0).mean():.4f}')
    for lab,a,b in [('2020-25','2020','2025-12-31'),('2026-28','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-07-10')]:
        x=q[(q.index>=a)&(q.index<=b)]
        if len(x): print(lab,len(x),f'IC={x.ic.mean():.6f}',f'ICIR={x.ic.mean()/x.ic.std(ddof=1):.6f}')
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20300711_consensus_trend_signal.csv',index=False)
print('symbols',out.symbol.nunique(),'rows',len(out),'coverage',len(out)/(len(f)*len(syms)),'turnover',f.rank(pct=True).diff().abs().mean().mean())

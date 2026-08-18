import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def main():
    acc=get_account_dict(); wl=acc.get('watch_list',[])
    syms=[s for s in U if not wl or s in wl]
    px={}
    for s in syms:
        d=get_stock_daily_data(s, days=5000)
        if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
    p=pd.DataFrame(px).sort_index().ffill()
    # macro observation is intentionally not traded; use VIX as regime input if available
    try:
        v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(p.index).ffill()
    except Exception:
        v=pd.Series(0.,index=p.index)
    r=p.pct_change()
    # Volatility-adjusted intermediate momentum, reversed during abrupt volatility expansion.
    mom=p.pct_change(20); vol=r.rolling(20).std()*np.sqrt(20)
    vshock=v.pct_change(10).rolling(5).mean()
    # continuous macro conditioning: momentum in calm/settling volatility, reversal in shock
    shock=np.clip(vshock/0.20, -2, 2)
    f=mom.div(vol.replace(0,np.nan)).mul(1-shock,axis=0)
    records=[]; horizons=[5,10,20,40]
    for i in range(100,len(p)-40):
        dt=p.index[i]; x=f.iloc[i]; n=x.notna().sum()
        if n<8: continue
        z=x.dropna(); row={'date':dt,'n':n}
        for h in horizons:
            y=p.iloc[i+h].div(p.iloc[i]).sub(1).reindex(z.index)
            row[f'ic{h}']=z.corr(y,method='spearman')
        records.append(row)
    q=pd.DataFrame(records)
    print('dates',len(q),'mean_n',q.n.mean(),'period',q.date.min(),q.date.max())
    for h in horizons:
        x=q[f'ic{h}'].dropna(); print(h,'IC %.6f ICIR %.6f hit %.4f'%(x.mean(),x.mean()/x.std(ddof=1), (x>0).mean()))
    # signal turnover and coverage, computed on valid factor panels
    valid=f.notna().sum(axis=1); print('coverage %.4f turnover_proxy %.4f'%(valid.mean()/len(syms), f.rank(axis=1,pct=True).diff().abs().mean().mean()))
    q.to_csv('../persistent/miner_1_20350216_macro_vol_conditioned_momentum_ic.csv',index=False)
    f.to_csv('../persistent/miner_1_20350216_macro_vol_conditioned_momentum_signal.csv')
if __name__=='__main__': main()

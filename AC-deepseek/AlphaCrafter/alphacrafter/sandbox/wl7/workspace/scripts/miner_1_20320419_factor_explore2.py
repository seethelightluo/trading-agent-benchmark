import pandas as pd, numpy as np

NAMES = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
VIS = '2032-04-19'

def load(n):
    df = pd.read_csv(f'../persistent/stock_data/{n}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[df['date'] <= VIS].set_index('date')

closes = {n: load(n)['close'].astype(float) for n in NAMES}
grid = pd.DatetimeIndex(sorted(set().union(*[c.index for c in closes.values()])))
R = pd.DataFrame({n: closes[n].reindex(grid).pct_change() for n in NAMES})
print("grid:", len(grid), grid[0].date(), "->", grid[-1].date())

F={}
# 1) price vs 200d MA (trend)
for n in NAMES:
    s=closes[n]
    F.setdefault('trend_200d',{})[n]=(s/s.rolling(200).mean()-1).reindex(grid)
# 2) 52w high proximity
for n in NAMES:
    s=closes[n]
    F.setdefault('high_prox_252',{})[n]=(s/s.rolling(252).max()).reindex(grid)
# 3) efficiency ratio 60d (trend vs noise)
for n in NAMES:
    p=closes[n].pct_change()
    er=closes[n].diff(60).abs()/(p.abs().rolling(60).sum()+1e-9)
    F.setdefault('eff_ratio_60',{})[n]=er.reindex(grid)
# 4) risk-adjusted momentum 20 (mom_20 / rv20)
for n in NAMES:
    s=closes[n]; p=s.pct_change()
    mom=s.shift(5)/s.shift(25)-1
    F.setdefault('sharpe_20',{})[n]=(mom/(p.rolling(20).std()+1e-9)).reindex(grid)
# 5) 5d momentum vs cross median (short cross momentum)
for n in NAMES:
    s=closes[n]
    F.setdefault('rel_mom_5d',{})[n]=(s.shift(1)/s.shift(6)-1).reindex(grid)
# 6) dispersion / risk parity moment: inverse-vol 20d
for n in NAMES:
    p=closes[n].pct_change()
    F.setdefault('inv_vol_20',{})[n]=(1.0/(p.rolling(20).std()+1e-9)).reindex(grid)
# 7) beta to SPX 60d (reuse concept but to latest)
spx=R['SPX']
for n in NAMES:
    b=(R[n].rolling(60).cov(spx)+1e-12)/(spx.rolling(60).var()+1e-12)
    F.setdefault('beta_spx_60',{})[n]=b.reindex(grid)
# 8) momentum relative to 60d base (acceleration)
for n in NAMES:
    s=closes[n]
    mom20=s.shift(5)/s.shift(25)-1
    mom60=s.shift(20)/s.shift(80)-1
    F.setdefault('mom_accel_20x60',{})[n]=(mom20-mom60).reindex(grid)

def test(fid, horizon=10, min_cov=8):
    FF=pd.DataFrame(F[fid])
    fwd=pd.DataFrame({n:closes[n].reindex(grid).shift(-horizon)/closes[n].reindex(grid)-1.0 for n in NAMES})
    ics=[];dates=[]
    for t in range(len(grid)-horizon):
        frow=FF.iloc[t];rrow=fwd.iloc[t]
        m=frow.notna()&rrow.notna()
        if m.sum()<min_cov: continue
        ic=frow[m].rank().corr(rrow[m].rank())
        if pd.notna(ic): ics.append(ic);dates.append(grid[t])
    ics=np.array(ics)
    ir=ics.mean()/ics.std() if len(ics)>1 and ics.std()>0 else 0.0
    d1=np.array([d>=np.datetime64('2031-04-19') for d in dates]); s1=ics[d1]
    ir1=s1.mean()/s1.std() if len(s1)>1 and s1.std()>0 else 0.0
    return ics.mean(),ir,s1.mean() if len(s1) else 0,ir1 if len(s1) else 0,len(ics),FF.notna().mean().mean()

print(f"\n{'factor':<20}{'fullIC':>8}{'fullICIR':>9}{'1yIC':>7}{'1yICIR':>8}{'n':>6}{'cov':>6}")
for fid in F:
    ic,ir,ic1,ir1,n,cov=test(fid)
    print(f"{fid:<20}{ic:>8.4f}{ir:>9.3f}{ic1:>7.4f}{ir1:>8.3f}{n:>6}{cov:>6.3f}")
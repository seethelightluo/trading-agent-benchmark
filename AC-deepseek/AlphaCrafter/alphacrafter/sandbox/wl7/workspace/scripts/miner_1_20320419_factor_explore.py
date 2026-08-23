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
vols   = {n: load(n)['volume'].astype(float) for n in NAMES}

grid = pd.DatetimeIndex(sorted(set().union(*[c.index for c in closes.values()])))
R = pd.DataFrame({n: closes[n].reindex(grid).pct_change() for n in NAMES})
print("grid dates:", len(grid), grid[0].date(), "->", grid[-1].date())

F = {}
# 1) skew 20d skip5
for n in NAMES:
    p = closes[n].pct_change()
    F.setdefault('skew_20d_skip5',{})[n] = p.shift(5).rolling(20).skew().reindex(grid)
# 2) drawdown 60d
for n in NAMES:
    s = closes[n]
    F.setdefault('drawdown_60d',{})[n] = ((s / s.rolling(60).max()) - 1.0).reindex(grid)
# 3) updown ratio 20
for n in NAMES:
    p = closes[n].pct_change()
    up = p.where(p>0,0.0); dn = p.where(p<0,0.0)
    F.setdefault('updown_ratio_20',{})[n] = (up.rolling(20).sum()/(-dn.rolling(20).sum()+1e-12)).reindex(grid)
# 4) mom 5d skip1
for n in NAMES:
    s = closes[n]
    F.setdefault('mom_5d_skip1',{})[n] = (s.shift(1)/s.shift(6)-1).reindex(grid)
# 5) mom 60d skip20
for n in NAMES:
    s = closes[n]
    F.setdefault('mom_60d_skip20',{})[n] = (s.shift(20)/s.shift(80)-1).reindex(grid)
# 6) vol ratio 20x60
for n in NAMES:
    p = closes[n].pct_change()
    F.setdefault('vol_ratio_20x60',{})[n] = (p.rolling(20).std()/p.rolling(60).std()).reindex(grid)
# 7) rel vs EW 20d
ow = R.mean(axis=1, skipna=True).rolling(20).mean().reindex(grid)
for n in NAMES:
    run = closes[n].pct_change().rolling(20).mean()
    F.setdefault('rel_vs_ew_20',{})[n] = (run - ow).reindex(grid)
# 8) avg true-ish dispersion: realized vol 40d level (cross-sectional predictability of vol)
for n in NAMES:
    F.setdefault('rv_40d',{})[n] = R[n].rolling(40).std().reindex(grid)

def test(fid, horizon=10, min_cov=8):
    FF = pd.DataFrame(F[fid])
    fwd = pd.DataFrame({n: closes[n].reindex(grid).shift(-horizon)/closes[n].reindex(grid)-1.0 for n in NAMES})
    ics=[]; dates=[]
    for t in range(len(grid)-horizon):
        frow=FF.iloc[t]; rrow=fwd.iloc[t]
        m=frow.notna()&rrow.notna()
        if m.sum()<min_cov: continue
        ic=frow[m].rank().corr(rrow[m].rank())
        if pd.notna(ic): ics.append(ic); dates.append(grid[t])
    ics=np.array(ics)
    cov = FF.notna().mean().mean()
    ir = ics.mean()/ics.std() if len(ics)>1 and ics.std()>0 else 0.0
    # recent 1y
    d1=np.array([d>=np.datetime64('2031-04-19') for d in dates])
    s1=ics[d1]
    ir1=s1.mean()/s1.std() if len(s1)>1 and s1.std()>0 else 0.0
    return ics.mean(), ir, s1.mean() if len(s1) else 0, ir1 if len(s1) else 0, len(ics), cov

print(f"\n{'factor':<18}{'fullIC':>8}{'fullICIR':>9}{'1yIC':>7}{'1yICIR':>8}{'n':>6}{'cov':>6}")
for fid in F:
    ic,ir,ic1,ir1,n,cov = test(fid)
    print(f"{fid:<18}{ic:>8.4f}{ir:>9.3f}{ic1:>7.4f}{ir1:>8.3f}{n:>6}{cov:>6.3f}")
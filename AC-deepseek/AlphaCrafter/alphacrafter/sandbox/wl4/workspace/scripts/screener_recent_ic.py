import pandas as pd, numpy as np, glob, os

files = sorted(glob.glob('../persistent/stock_data/*.csv'))
px = {}
for f in files:
    sym = os.path.basename(f)[:-4]
    df = pd.read_csv(f)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    px[sym] = df['close']
px = pd.DataFrame(px).dropna(how='all')
px = px.loc[px.index <= '2028-08-25'].sort_index()
rets = px.pct_change()

def rank_ic(factor_s, fwd, min_valid=8):
    out = []
    for t in factor_s.index:
        if t not in fwd.index: continue
        fs = factor_s.loc[t].dropna()
        fr = fwd.loc[t].dropna()
        common = fs.index.intersection(fr.index)
        if len(common) < min_valid: continue
        ic = np.corrcoef(fs[common].rank(), fr[common].rank())[0,1]
        if np.isnan(ic): continue
        out.append((t, ic))
    if not out: return pd.Series(dtype=float)
    s = pd.Series(dict(out)); return s

# forward 10d return
fwd10 = px.shift(-10)/px - 1.0

# Factor 1: vol_adj_mom_accel_20x60
mom20 = px/px.shift(20)-1
mom60 = px/px.shift(60)-1
vol20 = rets.rolling(20).std()
f1 = (mom20 - mom60)/vol20

# Factor 2: dn_mkt_beta_60d  (beta on down-market days)
mkt = rets.mean(axis=1)
down = mkt.where(mkt<0, 0.0)
def roll_beta(y, x, w=60, min_obs=40):
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    for col in y.columns:
        yy = y[col]; xx = x
        b = []
        for i in range(len(yy)):
            if i < w-1:
                b.append(np.nan); continue
            ys = yy.iloc[i-w+1:i+1]; xs = xx.iloc[i-w+1:i+1]
            m = ys.notna() & xs.notna()
            if m.sum() < min_obs:
                b.append(np.nan); continue
            if xs[m].var()==0:
                b.append(np.nan); continue
            b.append(np.polyfit(xs[m], ys[m], 1)[0])
        out[col] = b
    return out
f2 = roll_beta(rets, down, 60, 40)

# Factor 3: rate_beta_cn10y_60d (beta on CN10Y pct change)
cn10y_chg = px['CN10Y'].pct_change()
f3 = roll_beta(rets, cn10y_chg, 60, 40)

for name, f, exp_dir in [('vol_adj_mom_accel_20x60', f1, 1),
                          ('dn_mkt_beta_60d', f2, 1),
                          ('rate_beta_cn10y_60d', f3, -1)]:
    ic_all = rank_ic(f, fwd10)
    ic_2028 = rank_ic(f.loc['2028-01-01':], fwd10)
    ic_recent = rank_ic(f.loc['2028-05-01':], fwd10)
    ic_last60 = rank_ic(f.loc['2028-06-15':], fwd10)
    def stats(s):
        if len(s)==0: return (np.nan,np.nan,np.nan)
        return (s.mean(), s.mean()/s.std() if s.std()>0 else np.nan, (s>0).mean())
    a=stats(ic_all); b=stats(ic_2028); c=stats(ic_recent); d=stats(ic_last60)
    print(f'{name}  exp_dir={exp_dir:+d}')
    print(f'  full:    IC={a[0]:+.4f} ICIR={a[1]:+.3f} hit={a[2]:.3f} n={len(ic_all)}')
    print(f'  2028:    IC={b[0]:+.4f} ICIR={b[1]:+.3f} hit={b[2]:.3f} n={len(ic_2028)}')
    print(f'  since5/1:IC={c[0]:+.4f} ICIR={c[1]:+.3f} hit={c[2]:.3f} n={len(ic_recent)}')
    print(f'  last60:  IC={d[0]:+.4f} ICIR={d[1]:+.3f} hit={d[2]:.3f} n={len(ic_last60)}')

import pandas as pd, numpy as np, glob, os, json

# ---- load price data (visible through 2026-11-20) ----
files = sorted(glob.glob('../persistent/stock_data/*.csv'))
px = {}
for f in files:
    sym = os.path.basename(f).replace('.csv','')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    px[sym] = df['close'].astype(float)
px = pd.DataFrame(px)
px = px[px.index <= '2026-11-20'].sort_index()

def load_idx(sym):
    df = pd.read_csv(f'../persistent/index_data/{sym}.csv')
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').sort_index()['close'].astype(float)

eurusd = load_idx('EURUSD')

rets = px.pct_change()
mkt = rets.mean(axis=1)  # equal-weight 15-asset market

def rolling_beta(y, x, win=60, min_obs=40):
    out = pd.Series(index=y.index, dtype=float)
    for i in range(len(y)):
        if i < win: continue
        ys, xs = y.iloc[i-win:i], x.iloc[i-win:i]
        m = ys.notna() & xs.notna()
        if m.sum() < min_obs: continue
        out.iloc[i] = np.polyfit(xs[m], ys[m], 1)[0]
    return out

# volume
vol = {}
for f in files:
    sym = os.path.basename(f).replace('.csv','')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    vol[sym] = df['volume'].astype(float) if 'volume' in df.columns else np.nan
vol = pd.DataFrame(vol)
vol = vol[vol.index <= '2026-11-20'].sort_index()

f_vpc = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for c in px.columns:
    f_vpc[c] = rets[c].rolling(20).corr(vol[c])

f_eur = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for c in px.columns:
    f_eur[c] = rolling_beta(rets[c], eurusd.pct_change(), 60, 40)

f_cn = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for c in px.columns:
    f_cn[c] = rolling_beta(rets[c], px['CN10Y'].pct_change(), 60, 40)

f_dn = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
dnmkt = mkt.where(mkt < 0, 0.0)
for c in px.columns:
    f_dn[c] = rolling_beta(rets[c], dnmkt, 60, 40)

facs = {'vol_price_corr_20': f_vpc, 'eurusd_beta_60d': f_eur,
        'rate_beta_cn10y_60d': f_cn, 'dn_mkt_beta_60d': f_dn}

fwd = px.shift(-10)/px - 1

def rank_ic(fval, fwd10, start, end):
    fv = fval.loc[start:end]
    fr = fwd10.loc[start:end]
    ics = []
    for d in fv.index:
        x = fv.loc[d]; y = fr.loc[d]
        m = x.notna() & y.notna()
        if m.sum() < 6: continue
        ics.append((d, x[m].rank().corr(y[m].rank())))
    if not ics: return None
    return pd.Series(dict(ics))

for label, a, b in [('LIVE 08-03..11-09','2026-08-03','2026-11-09'),
                    ('LIVE 08-03..11-20','2026-08-03','2026-11-20')]:
    print(f'=== {label} (h=10 rank IC) ===')
    for name, fv in facs.items():
        s = rank_ic(fv, fwd, a, b)
        if s is None or len(s)==0:
            print(f'{name}: no data'); continue
        ic = s.mean(); icir = s.mean()/s.std() if s.std()>0 else np.nan
        hit = (np.sign(s)>0).mean()
        print(f'{name}: n={len(s):3d} IC={ic:+.4f} ICIR={icir:+.3f} hit={hit:.2f}')
    print()

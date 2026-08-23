"""Screener: compute recent (trailing 180d to 2028-12-13) rank-IC of candidate factors
for regime-aware selection. Uses price panel up to visible date."""
import pandas as pd, numpy as np, glob, os

files = sorted(glob.glob('../persistent/stock_data/*.csv'))
P = {}
for f in files:
    s = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    P[s] = df['close']
order = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
         'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
panel = pd.DataFrame(P).dropna(how='all').loc[:'2028-12-13'][order]
R = panel.pct_change().replace([np.inf, -np.inf], np.nan)


def rankz(x):
    x = x.astype(float)
    return x.rank(pct=True)


# ---------------- build factor panels (lightweight proxies) ----------------
F = {}
# momentum
F['mom_120d_skip5'] = panel / panel.shift(126) - 1
F['mom_10d_skip5'] = panel / panel.shift(15) - 1
# vol family
F['vol_z_20d'] = R.rolling(20).std() * np.sqrt(252)
F['bb_width_20d'] = F['vol_z_20d']
# autocorrelation
F['ac1_120d'] = R.rolling(120).apply(lambda x: x.autocorr() if len(x) == 120 and x.std() > 0 else np.nan, raw=False)
# skew
F['skew_20d'] = R.rolling(20).skew()
# kaufman efficiency
def kaufman(px, n=20):
    sig = px.diff(n).abs()
    noise = px.diff().abs().rolling(n).sum()
    return sig / noise
F['kaufman_eff_20d'] = kaufman(panel, 20)
# days since high
F['days_since_high_60'] = panel.rolling(60).apply(lambda x: (x == x.max()).sum() - 1 if len(x) == 60 else np.nan, raw=False)
# streak len 14 (sign of last day cumulative)
F['streak_len_14'] = panel.pct_change().rolling(14).apply(lambda x: (x > 0).sum() if len(x) == 14 else np.nan, raw=False)
# kurtosis
F['kurt_20d'] = R.rolling(20).kurt()
# range position
F['rng_pos_20d'] = (panel - panel.rolling(20).min()) / (panel.rolling(20).max() - panel.rolling(20).min())

# macro-beta style
vix = pd.read_csv('../persistent/index_data/VIX.csv')
vix.columns = [c.strip().lower() for c in vix.columns]
vix['date'] = pd.to_datetime(vix['date'])
vix = vix.sort_values('date').set_index('date')
vixc = vix['close']
rvix = vixc.pct_change()
def rolling_beta(y, x, n):
    out = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    for c in y.columns:
        yy = pd.concat([y[c], x], axis=1).dropna()
        b = yy.iloc[:, 0].rolling(n).cov(yy.iloc[:, 1]) / yy.iloc[:, 1].rolling(n).var()
        out[c] = b
    return out
F['beta_VIX_60'] = rolling_beta(R, rvix, 60)

cny = pd.read_csv('../persistent/index_data/USDCNY.csv')
cny.columns = [c.strip().lower() for c in cny.columns]
cny['date'] = pd.to_datetime(cny['date'])
cny = cny.sort_values('date').set_index('date')
rcny = cny.iloc[:, 0].pct_change()
F['cny_beta_60'] = rolling_beta(R, rcny, 60)


def rank_ic_series(fact, fwd, window=180):
    """Cross-sectional Spearman IC at each date vs 10d forward return."""
    dates = fact.index
    out = {}
    for d in dates:
        if d not in fwd.index:
            continue
        fr = fwd.loc[d]
        fv = fact.loc[d]
        mask = fv.notna() & fr.notna()
        n = mask.sum()
        if n < 8:
            continue
        ic = rankz(fv[mask]).corr(rankz(fr[mask]))
        out[d] = ic
    s = pd.Series(out).sort_index()
    return s

fwd10 = panel.shift(-10) / panel - 1  # forward 10d return at date t

print('=== trailing rank IC (180d window ending 2028-12-13) ===')
results = {}
for name, fv in F.items():
    fv = fv.replace([np.inf, -np.inf], np.nan)
    ic_s = rank_ic_series(fv, fwd10, 180)
    if len(ic_s) < 10:
        print(f'{name:24s} insufficient history ({len(ic_s)} obs)')
        results[name] = None
        continue
    recent = ic_s.tail(180)
    m = recent.mean()
    sd = recent.std()
    icir = m / sd if sd > 0 else np.nan
    hits = (recent > 0).mean()
    latest = ic_s.iloc[-1]
    results[name] = (m, icir, hits, latest)
    print(f'{name:24s} meanIC {m:+0.4f}  ICIR {icir:+0.3f}  hit {hits:0.3f}  lastIC {latest:+0.4f}  nobs {len(ic_s)}')

# correlation among top candidates (recent factor values)
print()
print('=== pairwise corr of recent factor exposures (latest) ===')
cand = ['beta_VIX_60', 'kaufman_eff_20d', 'mom_120d_skip5', 'mom_10d_skip5', 'bb_width_20d',
        'vol_z_20d', 'ac1_120d', 'skew_20d', 'cny_beta_60', 'days_since_high_60',
        'streak_len_14', 'kurt_20d', 'rng_pos_20d', 'mom_10_vixreg_dummy']
C = pd.DataFrame({k: v.iloc[-1] for k, v in F.items() if k in cand}).dropna(how='all')
C = C.dropna(axis=1, how='all')
corr = C.corr()
for i in range(len(corr)):
    for j in range(i + 1, len(corr)):
        a, b = corr.index[i], corr.columns[j]
        v = corr.iloc[i, j]
        if abs(v) > 0.6:
            print(f'{a:24s} x {b:24s} corr = {v:+.3f}')
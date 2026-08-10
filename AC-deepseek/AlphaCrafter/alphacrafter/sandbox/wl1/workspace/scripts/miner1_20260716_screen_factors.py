"""miner1 2026-07-16: screen broad battery of candidate factors on 15-asset panel."""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache.pkl')
close, high, low, opn, vol = panel['close'], panel['high'], panel['low'], panel['open'], panel['vol']
ret = panel['ret']
macro = panel['macro']
END = '2026-07-15'

def parkinson(h, l):
    return np.sqrt((np.log(h / l) ** 2).rolling(20).mean())

# ---------------- factor builders (higher = bullish signal) ----------------
F = {}
F['mom_20d'] = close / close.shift(20) - 1
F['mom_60d'] = close / close.shift(60) - 1
F['mom_120_20'] = close.shift(20) / close.shift(120) - 1          # 12-1 momentum
F['ramom_20d'] = F['mom_20d'] / ret.rolling(20).std()
F['ramom_60d'] = F['mom_60d'] / ret.rolling(60).std()
F['rev_1d'] = -ret
F['rev_5d'] = -(close / close.shift(5) - 1)
F['lowvol_20d'] = -ret.rolling(20).std()
F['lowvol_60d'] = -ret.rolling(60).std()
F['park_vol20'] = -parkinson(high, low)
F['vol_regime'] = -(ret.rolling(20).std() / ret.rolling(60).std() - 1)   # falling vol
F['dd_60d'] = close / close.rolling(60).max() - 1                        # drawdown depth (negative)
F['dd_120d'] = close / close.rolling(120).max() - 1
F['range_pos_20'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min())
F['skew_20d'] = ret.rolling(20).skew()
F['acorr_5d'] = ret.rolling(21).apply(lambda x: pd.Series(x).autocorr(5) if len(x) > 5 else np.nan, raw=False)
F['amihud_20'] = -(ret.abs() / vol).rolling(20).mean()
F['vol_z_60'] = -(ret.rolling(20).std() / ret.rolling(60).std().rolling(120).mean() - 1)

# cross-sectional market
mkt = ret.mean(axis=1)
mkt_ret = mkt

def beta_to(x, y, win=60):
    """rolling beta of x on y"""
    out = {}
    for a in x.columns:
        z = pd.concat([x[a].rename('x'), y.rename('y')], axis=1).dropna()
        cov = z['x'].rolling(win).cov(z['y'])
        var = z['y'].rolling(win).var()
        out[a] = cov / var
    return pd.DataFrame(out)

F['beta_mkt_60'] = -beta_to(ret, mkt_ret, 60)                            # low beta
F['beta_mkt_chg'] = -(beta_to(ret, mkt_ret, 20) - beta_to(ret, mkt_ret, 60))  # falling beta
# macro betas
vix_chg = macro['VIX'].pct_change().reindex(ret.index).ffill()
F['beta_vix_60'] = -beta_to(ret, vix_chg, 60)                           # defensive: -beta to VIX rise
dxy_chg = macro['DXY'].pct_change().reindex(ret.index).ffill()
F['beta_dxy_60'] = -beta_to(ret, dxy_chg, 60)                           # -beta to USD strength
us10y_chg = ret['US10Y'].rename('us10y') if 'US10Y' in ret else None
F['beta_us10y_60'] = -beta_to(ret, us10y_chg, 60)                       # -beta to yield rise
# gold beta (risk-off refuge)
xau_ret = ret['XAU']
F['beta_xau_60'] = beta_to(ret, xau_ret, 60)

# overnight vs intraday
prev_close = close.shift(1)
overnight = opn / prev_close - 1
intraday = close / opn - 1
F['overnight_20'] = overnight.rolling(20).mean()
F['intraday_20'] = intraday.rolling(20).mean()
F['overnight_intraday'] = overnight.rolling(20).mean() - intraday.rolling(20).mean()

# up/down volatility asymmetry
up_vol = ret.where(ret > 0).rolling(60).std()
dn_vol = ret.where(ret < 0).rolling(60).std()
F['vol_asym_60'] = -(up_vol / dn_vol - 1)   # higher up-vol vs down-vol bearish? sign to test

# ---------------- IC evaluation ----------------
def ranks_matrix(X):
    """cross-sectional ranks per date, normalized to [0,1]"""
    return X.rank(axis=1, pct=True)

def ic_series(fac, fwd, min_valid=8):
    """daily spearman IC between factor and forward return"""
    fac = fac.reindex(fwd.index)
    ic, dates = [], []
    for d in fwd.index:
        fv = fac.loc[d] if d in fac.index else None
        if fv is None:
            continue
        frow = fv.dropna()
        rrow = fwd.loc[d].reindex(frow.index).dropna()
        frow = frow.reindex(rrow.index)
        if len(frow) < min_valid:
            continue
        # spearman via ranks
        fr = frow.rank()
        rr = rrow.rank()
        if fr.std() == 0 or rr.std() == 0:
            continue
        c = np.corrcoef(fr, rr)[0, 1]
        if np.isfinite(c):
            ic.append(c); dates.append(d)
    return pd.Series(ic, index=dates)

def evaluate(name, fac, horizons=(1, 2, 3, 5, 10, 20), min_valid=8):
    res = {}
    out = {}
    for h in horizons:
        fwd = ret.shift(-h)
        s = ic_series(fac, fwd, min_valid)
        if len(s) < 120:
            out[h] = None
            continue
        m = float(s.mean()); sd = float(s.std(ddof=1))
        icir = m / sd if sd > 1e-12 else 0.0
        out[h] = dict(ic=m, icir=icir, hit=float((s > 0).mean()), n=int(len(s)))
    # coverage: fraction of asset-days with valid factor value in eval window
    eval_start = '2021-01-04'
    sub = fac.loc[fac.index >= eval_start]
    cov = float(sub.notna().mean().mean()) if len(sub) else 0.0
    # turnover over 10d: mean abs change of cross-sectional rank (normalized)
    rk = fac.rank(axis=1, pct=True)
    rk10 = rk - rk.shift(10)
    turn = float(rk10.abs().mean().mean()) if len(rk10) else np.nan
    res[name] = dict(horizons=out, coverage=round(cov, 4), turnover_10d=round(turn, 4))
    return res

results = {}
for name, fac in F.items():
    results.update(evaluate(name, fac))

# print report
print(f"{'factor':24s} {'IC1':>8s} {'ICIR1':>8s} {'hit1':>6s} {'n1':>5s} {'IC5':>8s} {'IC10':>8s} {'IC20':>8s} {'cov':>6s} {'turn10':>7s}")
rows = []
for name, r in results.items():
    h = r['horizons']
    g = h.get(1)
    if g is None:
        continue
    h5, h10, h20 = h.get(5), h.get(10), h.get(20)
    rows.append((name, g['ic'], g['icir'], g['hit'], g['n'],
                 h5['ic'] if h5 else np.nan, h10['ic'] if h10 else np.nan,
                 h20['ic'] if h20 else np.nan, r['coverage'], r['turnover_10d']))
rows.sort(key=lambda x: -abs(x[1]))
for r in rows:
    print(f"{r[0]:24s} {r[1]:8.4f} {r[2]:8.4f} {r[3]:6.3f} {r[4]:5d} {r[5]:8.4f} {r[6]:8.4f} {r[7]:8.4f} {r[8]:6.3f} {r[9]:7.3f}")

with open('scripts/miner1_screen_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner1_screen_results.json")

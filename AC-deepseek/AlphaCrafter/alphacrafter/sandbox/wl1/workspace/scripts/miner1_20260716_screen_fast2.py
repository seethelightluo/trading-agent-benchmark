"""miner1 2026-07-16: fully vectorized screening of candidate factors."""
import pandas as pd, numpy as np, json, time

t0 = time.time()
panel = pd.read_pickle('scripts/panel_cache.pkl')
close, high, low, opn, vol = panel['close'], panel['high'], panel['low'], panel['open'], panel['vol']
ret = panel['ret']
macro = panel['macro']

def roll_beta(x, y, win):
    ym = y.rolling(win).mean()
    xm = x.rolling(win).mean()
    cov = (x.mul(y, axis=0)).rolling(win).mean() - xm.mul(ym, axis=0)
    var = y.rolling(win).var()
    return cov.div(var, axis=0)

# ---------------- factor builders (higher = bullish signal) ----------------
F = {}
F['mom_20d'] = close / close.shift(20) - 1
F['mom_60d'] = close / close.shift(60) - 1
F['mom_120_20'] = close.shift(20) / close.shift(120) - 1
F['ramom_20d'] = F['mom_20d'] / ret.rolling(20).std()
F['ramom_60d'] = F['mom_60d'] / ret.rolling(60).std()
F['rev_1d'] = -ret
F['rev_5d'] = -(close / close.shift(5) - 1)
F['lowvol_20d'] = -ret.rolling(20).std()
F['lowvol_60d'] = -ret.rolling(60).std()
F['park_vol20'] = -np.sqrt((np.log(high / low) ** 2).rolling(20).mean())
F['vol_regime'] = -(ret.rolling(20).std() / ret.rolling(60).std() - 1)
F['dd_60d'] = close / close.rolling(60).max() - 1
F['dd_120d'] = close / close.rolling(120).max() - 1
F['range_pos_20'] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min())
F['skew_20d'] = ret.rolling(20).skew()
r5 = ret.shift(5)
cov5 = (ret.mul(r5)).rolling(21).mean() - ret.rolling(21).mean().mul(r5.rolling(21).mean(), axis=0)
den = ret.rolling(21).std().mul(r5.rolling(21).std(), axis=0)
F['acorr_5d'] = cov5 / den.replace(0, np.nan)
F['amihud_20'] = -(ret.abs() / vol).rolling(20).mean()
F['vol_z_60'] = -(ret.rolling(20).std() / ret.rolling(60).std().rolling(120).mean() - 1)

mkt_ret = ret.mean(axis=1)
F['beta_mkt_60'] = -roll_beta(ret, mkt_ret, 60)
F['beta_mkt_chg'] = -(roll_beta(ret, mkt_ret, 20) - roll_beta(ret, mkt_ret, 60))
vix_chg = macro['VIX'].pct_change().reindex(ret.index).ffill()
F['beta_vix_60'] = -roll_beta(ret, vix_chg, 60)
dxy_chg = macro['DXY'].pct_change().reindex(ret.index).ffill()
F['beta_dxy_60'] = -roll_beta(ret, dxy_chg, 60)
F['beta_us10y_60'] = -roll_beta(ret, ret['US10Y'], 60)
F['beta_xau_60'] = roll_beta(ret, ret['XAU'], 60)

prev_close = close.shift(1)
overnight = opn / prev_close - 1
intraday = close / opn - 1
F['overnight_20'] = overnight.rolling(20).mean()
F['intraday_20'] = intraday.rolling(20).mean()
F['overnight_intraday'] = overnight.rolling(20).mean() - intraday.rolling(20).mean()

up_vol = ret.where(ret > 0).rolling(60).std()
dn_vol = ret.where(ret < 0).rolling(60).std()
F['vol_asym_60'] = -(up_vol / dn_vol - 1)

print(f"factors built in {time.time()-t0:.1f}s")

def row_spearman(X, Y, min_valid=8):
    """Vectorized cross-sectional spearman IC per date. X,Y already rank-transformed DataFrames."""
    x = X.to_numpy(dtype=float)
    y = Y.to_numpy(dtype=float)
    valid = (~np.isnan(x)) & (~np.isnan(y))
    n = valid.sum(axis=1)
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    with np.errstate(all='ignore'):
        xm = np.nanmean(x, axis=1, keepdims=True)
        ym = np.nanmean(y, axis=1, keepdims=True)
        xc = np.where(valid, x - xm, 0.0)
        yc = np.where(valid, y - ym, 0.0)
        num = (xc * yc).sum(axis=1)
        dx = np.sqrt((xc * xc).sum(axis=1))
        dy = np.sqrt((yc * yc).sum(axis=1))
        corr = num / (dx * dy)
    corr = np.where((n >= min_valid) & np.isfinite(corr), corr, np.nan)
    return corr

HORIZONS = (1, 2, 3, 5, 10, 20)
# pre-rank forward returns once per horizon
fwd_ranks = {h: ret.shift(-h).rank(axis=1) for h in HORIZONS}

def evaluate(name, fac):
    out = {}
    fr = fac.rank(axis=1)
    for h in HORIZONS:
        s = row_spearman(fr, fwd_ranks[h])
        s = s[~np.isnan(s)]
        if len(s) < 120:
            out[h] = None
            continue
        m = float(s.mean()); sd = float(s.std(ddof=1))
        icir = m / sd if sd > 1e-12 else 0.0
        out[h] = dict(ic=m, icir=icir, hit=float((s > 0).mean()), n=int(len(s)))
    eval_start = '2021-01-04'
    sub = fac.loc[fac.index >= eval_start]
    cov = float(sub.notna().mean().mean()) if len(sub) else 0.0
    rk = fac.rank(axis=1, pct=True)
    turn = float((rk - rk.shift(10)).abs().mean().mean()) if len(rk) else np.nan
    return dict(horizons=out, coverage=round(cov, 4), turnover_10d=round(turn, 4))

results = {name: evaluate(name, fac) for name, fac in F.items()}

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
print(f"\nsaved scripts/miner1_screen_results.json in {time.time()-t0:.1f}s")

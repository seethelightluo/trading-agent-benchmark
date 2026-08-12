"""miner3 2029-08-17: re-validate effective factors through 2029-08-16 (vectorized IC)."""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache.pkl')
C = panel['close']; O = panel['open']; H = panel['high']; L = panel['low']
R = panel['ret']
gate_ic, gate_icir = 0.0070, 0.0840

factors = {}
factors["mom_120d_skip5"] = C.shift(5) / C.shift(125) - 1.0
for nd in (1, 2, 3, 5):
    factors[f"miner2_rev_{nd}d"] = -(C.shift(nd) / C - 1.0)
factors["miner2_rev_1d_vs"] = -(C.shift(1) / C - 1.0) * np.sign(C - C.shift(1))
for nd in (1, 2, 3, 5):
    hi = C.rolling(nd).max(); lo = C.rolling(nd).min()
    factors[f"miner2_nclv_{nd}d"] = (C - lo) / (hi - lo) - 0.5
factors["miner2_id_rev_1d"] = -(C / O - 1.0)
rng = (H - L).replace(0, np.nan)
factors["miner2_nbody_1d"] = (C - L) / rng - 0.5
factors["vol_of_vol20x60"] = R.rolling(20).std().rolling(60).std()
M = panel["macro"]
vix = M["VIX"]
vix_ret = vix.pct_change()
cov60 = R.rolling(60).cov(vix_ret)
var60 = vix_ret.rolling(60).var()
beta_vix = cov60 / var60
cond = (vix > vix.rolling(20).mean()).astype(float)
factors["vix_beta_cond_60x20"] = beta_vix * cond


def ic_series_vec(X, F):
    valid = X.notna() & F.notna()
    n = valid.sum(axis=1)
    keep = n >= 8
    Xr = X.rank(axis=1).where(valid)
    Fr = F.rank(axis=1).where(valid)
    dX = Xr.sub(Xr.mean(axis=1), axis=0)
    dF = Fr.sub(Fr.mean(axis=1), axis=0)
    num = (dX * dF).sum(axis=1)
    den = np.sqrt((dX ** 2).sum(axis=1) * (dF ** 2).sum(axis=1))
    ic = (num / den).where(den > 0)
    ic = ic[keep & ic.notna()]
    return ic


def stats(ic_ser):
    if len(ic_ser) == 0:
        return dict(n=0, ic=float('nan'), icir=float('nan'), hit=float('nan'), last=None)
    return dict(n=len(ic_ser), ic=float(ic_ser.mean()),
                icir=float(ic_ser.mean() / ic_ser.std()) if len(ic_ser) > 1 else float('nan'),
                hit=float((ic_ser > 0).mean()), last=ic_ser.index.max().date())


out = {}
print(f"{'factor':22s} {'h1 ic':>9s} {'icir':>7s} {'hit':>6s} {'n':>5s} | {'365d ic':>9s} {'icir':>7s} {'n':>5s} | {'120d ic':>9s} {'icir':>7s} {'n':>5s} | gate")
for name, f in factors.items():
    row = {}
    for h in (1, 2, 3, 5):
        fwd = C.shift(-h) / C - 1.0
        ic_ser = ic_series_vec(f, fwd)
        row[h] = stats(ic_ser)
    ic1 = ic_series_vec(f, C.shift(-1) / C - 1.0)
    out[name] = row
    if len(ic1):
        last365 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=365)]
        last120 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=120)]
        n365 = len(last365); n120 = len(last120)
        ic365 = last365.mean() if n365 else np.nan
        icir365 = last365.mean() / last365.std() if n365 > 1 else np.nan
        ic120 = last120.mean() if n120 else np.nan
        icir120 = last120.mean() / last120.std() if n120 > 1 else np.nan
    else:
        n365 = n120 = 0; ic365 = icir365 = ic120 = icir120 = np.nan
    ok = (abs(row[1]['ic']) >= gate_ic) and (abs(row[1]['icir']) >= gate_icir)
    print(f"{name:22s} {row[1]['ic']:+9.5f} {row[1]['icir']:+7.3f} {row[1]['hit']:6.3f} {row[1]['n']:5d} | "
          f"{ic365:+9.5f} {icir365:+7.3f} {n365:5d} | {ic120:+9.5f} {icir120:+7.3f} {n120:5d} | {'PASS' if ok else '--'}")

with open('scripts/miner3_20290817_revalidate_results.json', 'w') as f:
    json.dump(out, f, default=str, indent=1)
print("saved scripts/miner3_20290817_revalidate_results.json")

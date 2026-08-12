"""miner3 2029-07-20: explore novel factor candidates round 2 (reversal/OHLC/macro-beta variants)."""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache.pkl')
C = panel['close']; O = panel['open']; H = panel['high']; L = panel['low']
R = panel['ret']; V = panel['vol']
M = panel['macro']
gate_ic, gate_icir = 0.0070, 0.0840

def vol(nd):
    return R.rolling(nd).std()

v20 = vol(20)

factors = {}
# --- reversal family (historically strongest in this panel) ---
factors["rev_1d"] = -(C.shift(1) / C - 1.0)
factors["rev_1d_voladj"] = -(C.shift(1) / C - 1.0) / v20
factors["rev_5d_voladj"] = -(C.shift(5) / C - 1.0) / v20
factors["rev_1d_cond_vix"] = -(C.shift(1) / C - 1.0) * (M['VIX'] > M['VIX'].rolling(20).mean()).astype(float)
factors["rev_1d_cond_trend"] = -(C.shift(1) / C - 1.0) * np.sign(C - C.rolling(20).mean())
# intraday reversal (close vs open) variants
factors["id_rev_1d"] = -(C / O - 1.0)
factors["id_rev_1d_voladj"] = -(C / O - 1.0) / v20
factors["id_rev_2d"] = -((C / O - 1.0) + (C.shift(1) / O.shift(1) - 1.0)) / 2.0
factors["id_rev_5d"] = -((C / O - 1.0) + (C.shift(1) / O.shift(1) - 1.0) + (C.shift(2) / O.shift(2) - 1.0) +
                        (C.shift(3) / O.shift(3) - 1.0) + (C.shift(4) / O.shift(4) - 1.0)) / 5.0
# OHLC range/body factors
rng = (H - L).replace(0, np.nan)
factors["nbody_1d"] = (C - L) / rng - 0.5
factors["upper_wick_1d"] = (H - np.maximum(O, C)) / rng
factors["lower_wick_1d"] = (np.minimum(O, C) - L) / rng
factors["wick_asym_1d"] = (H - np.maximum(O, C)) / rng - (np.minimum(O, C) - L) / rng
factors["range_exp_1d"] = (H - L) / (H.shift(1) - L.shift(1)) - 1.0
# overnight gap factor
factors["gap_1d"] = O / C.shift(1) - 1.0
# macro-beta variants (fixed construction)
us10y = C['US10Y']; d_us10y = us10y.diff()
cov60 = R.rolling(60).cov(d_us10y)
var60 = d_us10y.rolling(60).var().rename('u10yvar')
factors["us10y_beta_60"] = cov60.div(var60, axis=0) if False else cov60 / var60.values.reshape(-1, 1)
vix = M['VIX']; vix_ret = vix.pct_change()
cov60v = R.rolling(60).cov(vix_ret)
var60v = vix_ret.rolling(60).var()
factors["vix_beta_60"] = cov60v / var60v.values.reshape(-1, 1)
# conditional VIX beta on level regime (differs from existing change-conditioned factor)
cond_lvl = (vix > vix.rolling(20).mean()).astype(float)
factors["vix_beta_lvl_60x20"] = (cov60v / var60v.values.reshape(-1, 1)) * cond_lvl.values.reshape(-1, 1)
# VIX level z-score applied to all assets (risk-off intensity)
factors["vix_z_20"] = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()

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
    return ic[keep & ic.notna()]

def stats(ic_ser):
    if len(ic_ser) == 0:
        return dict(n=0, ic=float('nan'), icir=float('nan'), hit=float('nan'))
    return dict(n=len(ic_ser), ic=float(ic_ser.mean()),
                icir=float(ic_ser.mean() / ic_ser.std()) if len(ic_ser) > 1 else float('nan'),
                hit=float((ic_ser > 0).mean()))

print(f"{'factor':22s} {'h1 ic':>8s} {'icir':>7s} {'hit':>5s} {'n':>5s} | {'365d ic':>8s} {'icir':>7s} | {'120d ic':>8s} {'icir':>7s} | gate")
out = {}
for name, f in factors.items():
    if f is None:
        continue
    row = {}
    for h in (1, 2, 3, 5):
        fwd = C.shift(-h) / C - 1.0
        row[h] = stats(ic_series_vec(f, fwd))
    ic1 = ic_series_vec(f, C.shift(-1) / C - 1.0)
    out[name] = row
    if len(ic1):
        last365 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=365)]
        last120 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=120)]
        ic365 = last365.mean() if len(last365) else np.nan
        icir365 = last365.mean() / last365.std() if len(last365) > 1 else np.nan
        ic120 = last120.mean() if len(last120) else np.nan
        icir120 = last120.mean() / last120.std() if len(last120) > 1 else np.nan
    else:
        ic365 = icir365 = ic120 = icir120 = np.nan
    ok = (abs(row[1]['ic']) >= gate_ic) and (abs(row[1]['icir']) >= gate_icir)
    print(f"{name:22s} {row[1]['ic']:+8.5f} {row[1]['icir']:+7.4f} {row[1]['hit']:5.3f} {row[1]['n']:5d} | "
          f"{ic365:+8.5f} {icir365:+7.4f} | {ic120:+8.5f} {icir120:+7.4f} | {ok}")

json.dump(out, open("scripts/miner3_20290720_screen2_results.json", "w"), indent=1, default=str)
print("\nsaved scripts/miner3_20290720_screen2_results.json")

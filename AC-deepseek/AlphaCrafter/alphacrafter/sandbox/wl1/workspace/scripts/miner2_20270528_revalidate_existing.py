"""miner2 2027-05-28: re-validate existing factor library on fresh panel through 2027-05-27.
Admission gate: abs IC1 >= 0.0070 and abs ICIR1 >= 0.0840 (daily cross-sectional Spearman IC, 1d fwd).
Report full-window, recent-250d, and 2026-10+ windows.
"""
import pandas as pd
import numpy as np
import pickle
from scipy.stats import spearmanr

panel = pickle.load(open("scripts/miner2_panel.pkl", "rb"))
C, O, H, L, V, R, M = panel["close"], panel["open"], panel["high"], panel["low"], panel["vol"], panel["ret"], panel["macro"]
ASSETS = list(C.columns)
MIN_VALID = 8
lnC = np.log(C)
STUDY_START = "2021-01-01"


def daily_ic(fdf, h):
    fr = R.shift(-h)
    ic = []
    for i in range(len(fdf)):
        fv = fdf.iloc[i].values
        rv = fr.iloc[i].values
        m = np.isfinite(fv) & np.isfinite(rv)
        if m.sum() < MIN_VALID:
            continue
        rho = spearmanr(fv[m], rv[m]).correlation
        ic.append(rho if np.isfinite(rho) else np.nan)
    return np.array(ic)


def stats(ic, fdf):
    ok = np.isfinite(ic)
    if ok.sum() < 30:
        return dict(n=int(ok.sum()), ic=np.nan, icir=np.nan, hit=np.nan, ic5=np.nan, ic10=np.nan,
                    cov=np.nan, turn=np.nan)
    ic = ic[ok]
    m = np.nanmean(ic); s = np.nanstd(ic)
    ic5 = np.nanmean(daily_ic(fdf, 5)); ic10 = np.nanmean(daily_ic(fdf, 10))
    rk = fdf.rank(axis=1)
    turn = float(np.nanmean(np.abs(rk.diff()).values))
    cov = float(np.nanmean(fdf.notna().sum(axis=1) >= MIN_VALID))
    return dict(n=int(ok.sum()), ic=m, icir=m / s if s > 0 else np.nan,
                hit=float(np.mean(np.sign(ic) == np.sign(m))), ic5=ic5, ic10=ic10, cov=cov, turn=turn)


def build(fn):
    fdf = pd.DataFrame(index=C.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        fdf[a] = fn(a)
    return fdf


factors = {
    "rev_1d":      lambda a: -(lnC[a].diff(1)),
    "rev_2d":      lambda a: -(lnC[a].diff(2)),
    "rev_3d":      lambda a: -(lnC[a].diff(3)),
    "rev_5d":      lambda a: -(lnC[a].diff(5)),
    "nclv_1d":     lambda a: -(C[a] - L[a]) / (H[a] - L[a]),
    "nclv_2d":     lambda a: -(C[a] - L[a].rolling(2).min()) / (H[a].rolling(2).max() - L[a].rolling(2).min()),
    "nclv_3d":     lambda a: -(C[a] - L[a].rolling(3).min()) / (H[a].rolling(3).max() - L[a].rolling(3).min()),
    "nclv_5d":     lambda a: -(C[a] - L[a].rolling(5).min()) / (H[a].rolling(5).max() - L[a].rolling(5).min()),
    "nbody_1d":    lambda a: -(C[a] - O[a]) / (H[a] - L[a]),
    "id_rev_1d":   lambda a: -(C[a] / O[a] - 1.0),
    "rev_1d_vs":   lambda a: -(lnC[a].diff(1)) / R[a].rolling(20).std(),
    "mom_120d_s5": lambda a: C[a].shift(5) / C[a].shift(125) - 1.0,
    "vov_20x60":   lambda a: R[a].rolling(20).std().rolling(60).std(),
    "vix_beta_60x20": lambda a: None,
}

vb = pd.DataFrame(index=C.index, columns=ASSETS, dtype=float)
vix = M["VIX"]
vix_r = vix.pct_change()
for a in ASSETS:
    r = R[a]
    cov = r.rolling(60).cov(vix_r)
    var = vix_r.rolling(60).var()
    beta = cov / var
    vix_up = (vix_r.rolling(20).mean() > 0).astype(float)
    vb[a] = -beta * vix_up

rows = []
for fid, fn in factors.items():
    if fid == "vix_beta_60x20":
        fdf = vb
    else:
        fdf = build(fn).loc[STUDY_START:]
    ic1 = daily_ic(fdf, 1)
    s_full = stats(ic1, fdf)
    rec = fdf.iloc[-250:]
    ic1r = daily_ic(rec, 1)
    s_rec = stats(ic1r, rec)
    w27 = fdf.loc["2026-10-01":]
    ic1w = daily_ic(w27, 1)
    s_w27 = stats(ic1w, w27)
    rows.append(dict(factor=fid, **{f"f_{k}": v for k, v in s_full.items()},
                     **{f"r_{k}": v for k, v in s_rec.items()},
                     **{f"w_{k}": v for k, v in s_w27.items()}))

out = pd.DataFrame(rows)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
print("=== RE-VALIDATION: existing factor library (daily 1d-forward Spearman IC, 15 assets) ===")
print("full window 2021-01-01..2027-05-27 | recent = last 250d | w27 = 2026-10-01..")
cols = ["factor", "f_ic", "f_icir", "f_hit", "f_n", "f_ic5", "f_ic10", "f_cov", "f_turn",
        "r_ic", "r_icir", "r_n", "w_ic", "w_icir", "w_n"]
print(out[cols].to_string(index=False))

print("\n=== GATE CHECK (|IC|>=0.0070, |ICIR|>=0.0840) ===")
for _, r in out.iterrows():
    gf = abs(r["f_ic"]) >= 0.0070 and abs(r["f_icir"]) >= 0.0840
    gr = abs(r["r_ic"]) >= 0.0070 and abs(r["r_icir"]) >= 0.0840
    gw = abs(r["w_ic"]) >= 0.0070 and abs(r["w_icir"]) >= 0.0840
    print(f"{r['factor']:<14} full={'PASS' if gf else 'fail'}(ic={r['f_ic']:+.4f},icir={r['f_icir']:+.4f},n={int(r['f_n'])}) "
          f"recent={'PASS' if gr else 'fail'}(ic={r['r_ic']:+.4f},icir={r['r_icir']:+.4f},n={int(r['r_n'])}) "
          f"w27={'PASS' if gw else 'fail'}(ic={r['w_ic']:+.4f},icir={r['w_icir']:+.4f},n={int(r['w_n'])})")

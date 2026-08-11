"""miner2 2027-04-30: screen a battery of candidate factors on the fresh panel.
Admission gate: abs IC1 >= 0.0070 and abs ICIR1 >= 0.0840 (daily cross-sectional Spearman IC, 1d fwd).
Report full-window, recent-250d, and 2026-10+ windows."""
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


def daily_ic(fdf, h=1):
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
        return dict(n=int(ok.sum()), ic=np.nan, icir=np.nan, hit=np.nan, cov=np.nan, turn=np.nan)
    ic = ic[ok]
    m = np.nanmean(ic); s = np.nanstd(ic)
    rk = fdf.rank(axis=1)
    turn = float(np.nanmean(np.abs(rk.diff()).values))
    cov = float(np.nanmean(fdf.notna().sum(axis=1) >= MIN_VALID))
    return dict(n=int(ok.sum()), ic=m, icir=m / s if s > 0 else np.nan,
                hit=float(np.mean(np.sign(ic) == np.sign(m))), cov=cov, turn=turn)


def eval_factor(fdf, name):
    fdf = fdf.loc[STUDY_START:]
    s_full = stats(daily_ic(fdf, 1), fdf)
    rec = fdf.iloc[-250:]
    s_rec = stats(daily_ic(rec, 1), rec)
    w27 = fdf.loc["2026-10-01":]
    s_w = stats(daily_ic(w27, 1), w27)
    return dict(name=name, **{f"f_{k}": v for k, v in s_full.items()},
                **{f"r_{k}": v for k, v in s_rec.items()},
                **{f"w_{k}": v for k, v in s_w.items()})


def build(fn):
    fdf = pd.DataFrame(index=C.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        fdf[a] = fn(a)
    return fdf


cands = {}
# --- momentum / trend family ---
for n in [10, 20, 40, 60, 90, 120, 180, 250]:
    for sk in [1, 5, 10]:
        cands[f"mom_{n}d_skip{sk}"] = lambda a, n=n, sk=sk: C[a].shift(sk) / C[a].shift(sk + n) - 1.0
# risk-adjusted momentum (return / vol over same window)
for n in [60, 90, 120, 180]:
    for sk in [5, 10]:
        def f_rmom(a, n=n, sk=sk):
            r = R[a]
            num = np.log(C[a].shift(sk) / C[a].shift(sk + n))
            vol = r.rolling(n).std() * np.sqrt(252)
            return num / (vol.shift(sk) + 1e-9)
        cands[f"rmom_{n}d_skip{sk}"] = f_rmom
# --- drawdown / high-water mark family ---
for n in [60, 120, 250]:
    cands[f"dist_hi_{n}d"] = lambda a, n=n: C[a] / C[a].rolling(n).max() - 1.0
cands["dd_250d"] = lambda a: C[a] / C[a].rolling(250).max() - 1.0
# --- volatility family ---
for n in [10, 20, 40, 60]:
    cands[f"vol_{n}d"] = lambda a, n=n: -R[a].rolling(n).std()
cands["vol_z_60x20"] = lambda a: -(R[a].rolling(20).std() - R[a].rolling(60).std()) / (R[a].rolling(60).std() + 1e-9)
cands["vol_of_vol_20x60"] = lambda a: R[a].rolling(20).std().rolling(60).std()
# --- volume family ---
cands["vol_ratio_5_20"] = lambda a: V[a].rolling(5).mean() / (V[a].rolling(20).mean() + 1e-9)
cands["vol_ratio_10_60"] = lambda a: V[a].rolling(10).mean() / (V[a].rolling(60).mean() + 1e-9)
cands["vol_trend_20"] = lambda a: V[a].rolling(5).mean() / (V[a].rolling(20).mean() + 1e-9) - 1.0
# --- range / location family ---
for n in [5, 10, 20]:
    cands[f"loc_{n}d"] = lambda a, n=n: (C[a] - L[a].rolling(n).min()) / (H[a].rolling(n).max() - L[a].rolling(n).min() + 1e-9)
for n in [5, 10, 20]:
    cands[f"nloc_{n}d"] = lambda a, n=n: -(C[a] - L[a].rolling(n).min()) / (H[a].rolling(n).max() - L[a].rolling(n).min() + 1e-9)
# --- combined reversal + volume confirmation ---
cands["rev1_x_vol_ratio"] = lambda a: -(lnC[a].diff(1)) * (V[a].rolling(5).mean() / (V[a].rolling(20).mean() + 1e-9))
# --- macro-conditioned ---
dxy_r = M["DXY"].pct_change()
usdjpy_r = M["USDJPY"].pct_change()
eur_r = M["EURUSD"].pct_change()

def beta_to_macro(a, macro_r, w=60):
    return R[a].rolling(w).cov(macro_r) / (macro_r.rolling(w).var() + 1e-12)

for w in [40, 60, 90]:
    cands[f"dxy_beta_{w}"] = lambda a, w=w: -beta_to_macro(a, dxy_r, w)
    cands[f"jpy_beta_{w}"] = lambda a, w=w: beta_to_macro(a, usdjpy_r, w)
    cands[f"eur_beta_{w}"] = lambda a, w=w: beta_to_macro(a, eur_r, w)
# --- volatility-scaled reversal ---
for n in [1, 2, 5]:
    cands[f"rev{n}_vs20"] = lambda a, n=n: -(lnC[a].diff(n)) / (R[a].rolling(20).std() + 1e-9)

# --- trend strength: count of up days ---
for n in [20, 60]:
    cands[f"updays_{n}d"] = lambda a, n=n: (R[a].rolling(n).apply(lambda x: (x > 0).sum(), raw=True)) / n

print(f"total candidates: {len(cands)}")
results = []
for name, fn in cands.items():
    try:
        fdf = build(fn)
        results.append(eval_factor(fdf, name))
    except Exception as e:
        print(f"ERROR {name}: {e}")

out = pd.DataFrame(results)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
print("\n=== CANDIDATE SCREEN (sorted by full-window |ICIR|) ===")
out["abs_icir"] = out["f_icir"].abs()
out = out.sort_values("abs_icir", ascending=False)
cols = ["name", "f_ic", "f_icir", "f_hit", "f_n", "f_cov", "f_turn", "r_ic", "r_icir", "r_n", "w_ic", "w_icir", "w_n"]
print(out[cols].head(45).to_string(index=False))

print("\n=== GATE CHECK (full-window |IC|>=0.007, |ICIR|>=0.084; also show recent/w27) ===")
for _, r in out.iterrows():
    gf = abs(r["f_ic"]) >= 0.0070 and abs(r["f_icir"]) >= 0.0840
    gr = abs(r["r_ic"]) >= 0.0070 and abs(r["r_icir"]) >= 0.0840
    gw = abs(r["w_ic"]) >= 0.0070 and abs(r["w_icir"]) >= 0.0840
    if gf or gr or gw:
        print(f"{r['name']:<22} full={'PASS' if gf else 'fail'}(ic={r['f_ic']:+.4f},icir={r['f_icir']:+.4f},n={int(r['f_n'])}) "
              f"recent={'PASS' if gr else 'fail'}(ic={r['r_ic']:+.4f},icir={r['r_icir']:+.4f},n={int(r['r_n'])}) "
              f"w27={'PASS' if gw else 'fail'}(ic={r['w_ic']:+.4f},icir={r['w_icir']:+.4f},n={int(r['w_n'])})")

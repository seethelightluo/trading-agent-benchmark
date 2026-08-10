"""miner_1: screen NOVEL factor families (macro-beta, return-shape, liquidity,
risk-adjusted momentum) against the CURRENT effective library.

Gate (15-name cross-asset universe, daily rank IC): |IC1| >= 0.0070, |ICIR1| >= 0.0840.
Validation window: 2021-01-01 .. 2026-07-15 (2020 as warm-up for rolling windows).
Library correlation: reconstructed from real artifacts (.npy / embedded) where
available, else from documented definitions - used as diversity provenance only.
"""
import sys, os, time, json, base64, gzip
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import miner3_fast as F

T0 = time.time()
SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
GATE_IC, GATE_ICIR = 0.0070, 0.0840
VALID0 = pd.Timestamp("2021-01-01")
CUT = pd.Timestamp("2026-07-15")

panel = pd.read_pickle("scripts/panel_cache.pkl")
CP, OP, HP, LP, VLP = (panel[k] for k in ("close", "open", "high", "low", "vol"))
MACRO = panel["macro"]
idx = CP.index
VAL = idx[(idx >= VALID0) & (idx <= CUT)]
print(f"panel {idx.min().date()}..{idx.max().date()} rows={len(idx)} "
      f"val rows={len(VAL)} [{time.time()-T0:.1f}s]")

LOG = np.log(CP / CP.shift(1))
RET = CP.pct_change()

# ---- macro returns aligned to panel index ----
MAC = {}
for c in MACRO.columns:
    s = MACRO[c].reindex(idx)
    MAC[c] = np.log(s / s.shift(1))
MACC = pd.DataFrame(MAC)
fwd = {h: F.fwd_returns({s: panel["close"][s] for s in SYMBOLS}, h).reindex(idx)
       for h in (1, 2, 3, 5, 10, 20, 30)}
N_CELLS = len(VAL) * len(SYMBOLS)


def roll_beta(asset_ret, mkt_ret, win, minp=None):
    """rolling beta of asset_ret vs mkt_ret over win days"""
    minp = minp or int(win * 0.6)
    df = pd.DataFrame({"a": asset_ret, "m": mkt_ret})
    cov = df["a"].rolling(win, min_periods=minp).cov(df["m"])
    var = df["m"].rolling(win, min_periods=minp).var()
    return (cov / var.replace(0, np.nan)).reindex(idx)


def lib_panels():
    """Reconstruct current library factor panels (dates x symbols) for rho."""
    lib = {}
    # artifacts: miner2 npy (2388x15 full index)
    for tag, fn in [("mom10s5", "factors/miner2_20260716_mom_10d_skip5.npy"),
                    ("nclv1", "factors/miner2_20260716_nclv_1d.npy")]:
        if os.path.exists(fn):
            M = np.load(fn, allow_pickle=True)
            lib[tag] = pd.DataFrame(M, index=idx, columns=SYMBOLS)
    # embedded artifacts (miner3)
    for fjson, tag in [("factors/miner3_20260716_rev_intraday_1d.json", "rev_intra1"),
                       ("factors/miner3_20260716_rev_intra_x_volrank.json", "rev_intra_x_vr"),
                       ("factors/miner3_20260716_volz_20.json", "volz20")]:
        if os.path.exists(fjson):
            d = json.load(open(fjson))
            art = d.get("signal_artifact")
            if isinstance(art, dict) and art.get("data"):
                try:
                    raw = base64.b64decode(art["data"])
                    dec = gzip.decompress(raw)
                    arr = np.load(io_bytes(dec))
                    lib[tag] = pd.DataFrame(arr, index=idx[idx >= VALID0], columns=art["symbols"])
                except Exception as e:
                    print(f"  [lib decode fail {tag}] {e}")
    # definitions
    hmax5, lmin5 = HP.rolling(5).max(), LP.rolling(5).min()
    lib["rev1"] = -LOG
    lib["clv5"] = -(CP - lmin5) / (hmax5 - lmin5).replace(0, np.nan)
    lib["mom120s5"] = np.log(CP.shift(5) / CP.shift(125))
    lib["volofvol"] = RET.rolling(20).std().rolling(60).std()
    vixr = MAC["VIX"]
    betavix = roll_beta(RET, vixr, 60)
    lib["vixbeta_cond"] = -betavix * (MACRO["VIX"].reindex(idx) / MACRO["VIX"].reindex(idx).shift(20) - 1.0)
    return lib


def io_bytes(b):
    import io
    return io.BytesIO(b)


LIB = lib_panels()
print(f"library panels reconstructed: {list(LIB.keys())} [{time.time()-T0:.1f}s]")


def panel_corr(a, b):
    A = a.values.astype(float)
    B = b.values.astype(float)
    m = np.isfinite(A) & np.isfinite(B)
    if int(m.sum()) < 100:
        return np.nan
    x, y = A[m], B[m]
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def run(name, p):
    p = p.reindex(idx)
    cov = float(p.reindex(VAL).notna().sum().sum()) / N_CELLS
    to = F.turnover10(p)
    ic1 = F.fast_ic(p, fwd[1])
    ics = {h: F.fast_ic(p, fwd[h]) for h in (1, 2, 3, 5, 10, 20, 30)}
    passed = (abs(ic1["ic"]) >= GATE_IC) and (abs(ic1["icir"]) >= GATE_ICIR)
    corrs = [panel_corr(p, lv) for lv in LIB.values()]
    corrs = [c for c in corrs if c is not None and np.isfinite(c)]
    maxc = max(abs(c) for c in corrs) if corrs else np.nan
    dec = " ".join(f"h{h}:{ics[h]['ic']:+.3f}" for h in (2, 3, 5, 10, 20))
    flag = "PASS" if passed else "fail"
    print(f"{name:20s} cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} "
          f"ICIR1={ic1['icir']:+.3f} hit1={ic1['hit']:.2f} n1={ic1['n_dates']:4d} "
          f"| maxLibRho={maxc:.2f} | {dec} | {flag}")
    return {"name": name, "panel": p, "cov": cov, "to": to, "ic": ics,
            "ic1": ic1, "passed": passed, "max_lib_corr": maxc}


# ===================== candidate construction =====================
cands = {}

# ---- Family A: macro beta (dollar / yen / euro / yuan / vix sensitivities) ----
for win in (20, 60):
    cands[f"dxy_beta_{win}"] = roll_beta(RET, MAC["DXY"], win)
    cands[f"jpy_beta_{win}"] = roll_beta(RET, MAC["USDJPY"], win)
    cands[f"eur_beta_{win}"] = roll_beta(RET, MAC["EURUSD"], win)
cands["cny_beta_60"] = roll_beta(RET, MAC["USDCNY"], 60)
cands["vix_beta_20"] = roll_beta(RET, MAC["VIX"], 20)

# ---- Family B: return-shape ----
cands["skew_20"] = RET.rolling(20).skew()
cands["skew_60"] = RET.rolling(60).skew()
def downside_ratio(win=60):
    dev = RET - RET.rolling(win).mean()
    sd = dev[dev < 0].rolling(win).apply(lambda x: np.sqrt((x ** 2).mean()), raw=True)
    return sd / (RET.rolling(win).std() + 1e-12)
cands["down_ratio_60"] = downside_ratio(60)
def auto1(win=60):
    out = {}
    for s in SYMBOLS:
        r = RET[s]
        out[s] = r.rolling(win).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1]
                                      if len(x) > 3 and x[:-1].std() > 0 and x[1:].std() > 0 else np.nan, raw=True)
    return pd.DataFrame(out)
cands["auto_60"] = auto1(60)

# ---- Family C: risk-adjusted momentum ----
cands["sharpe_20"] = RET.rolling(20).mean() / (RET.rolling(20).std() + 1e-12)
cands["sharpe_60"] = RET.rolling(60).mean() / (RET.rolling(60).std() + 1e-12)

# ---- Family D: liquidity / volume ----
absret = RET.abs()
cands["amihud_20"] = (absret / (VLP + 1e-9)).rolling(20).mean()
cands["vol_trend_5_60"] = np.log(VLP.rolling(5).mean() / VLP.rolling(60).mean())
cands["vol_disp_20"] = VLP.rolling(20).std() / (VLP.rolling(20).mean() + 1e-9)

# ---- Family E: level/position variants ----
hmax60, lmin60 = HP.rolling(60).max(), LP.rolling(60).min()
cands["dist_high_60"] = CP / hmax60 - 1.0
cands["dist_low_60"] = CP / lmin60 - 1.0
cands["clv_20"] = (CP - LP.rolling(20).min()) / (HP.rolling(20).max() - LP.rolling(20).min()).replace(0, np.nan)
cands["range_ratio_20"] = (HP - LP).rolling(20).mean() / CP

res = {}
for nm, p in cands.items():
    try:
        res[nm] = run(nm, p)
    except Exception as e:
        print(f"{nm}: ERROR {e}")

passers = {k: v for k, v in res.items() if v["passed"]}
print(f"\nTotal candidates: {len(cands)}, PASS: {len(passers)} -> {list(passers.keys())}")

# pairwise rho among passers
pn = list(passers.keys())
rho = {}
for i in range(len(pn)):
    for j in range(i + 1, len(pn)):
        r = panel_corr(passers[pn[i]]["panel"], passers[pn[j]]["panel"])
        rho[tuple(sorted((pn[i], pn[j])))] = r
        print(f"  rho {pn[i]:20s} | {pn[j]:20s} = {r:+.3f}")

# diversity selection: max_lib_corr < 0.5 and pairwise rho < 0.6, prefer by |ICIR|
pool = [nm for nm in passers if passers[nm]["max_lib_corr"] < 0.50]
print(f"\npassers with maxLibRho<0.50: {pool}")
selected = []
for nm in sorted(pool, key=lambda k: -abs(passers[k]["ic1"]["icir"])):
    if all(abs(rho[tuple(sorted((nm, s)))]) < 0.60 for s in selected):
        selected.append(nm)
print(f"Diverse selection: {selected}")

# by-year IC1 for selected
extra = {}
for nm in selected:
    p = passers[nm]["panel"]
    yr = {}
    for y in range(2021, 2027):
        m = (idx >= pd.Timestamp(f"{y}-01-01")) & (idx <= pd.Timestamp(f"{y}-12-31"))
        r = F.fast_ic(p.reindex(idx[m]), fwd[1].reindex(idx[m]))
        yr[y] = {"ic": round(r["ic"], 4), "icir": round(r["icir"], 3), "n": r["n_dates"]}
    extra[nm] = {"by_year": yr}
    print(f"{nm:20s} by_year={yr}")

# persist screening results for the persist step
import pickle
with open("scripts/_miner1_screen_v3.pkl", "wb") as fh:
    pickle.dump({nm: {"panel": passers[nm]["panel"], "cov": passers[nm]["cov"],
                      "to": passers[nm]["to"], "ic": passers[nm]["ic"],
                      "max_lib_corr": passers[nm]["max_lib_corr"], "passed": True}
                 for nm in selected}, fh)
print(f"\nfinished {time.time()-T0:.1f}s | candidates={len(cands)} pass={len(passers)} "
      f"selected={len(selected)} persisted_candidates={selected}")

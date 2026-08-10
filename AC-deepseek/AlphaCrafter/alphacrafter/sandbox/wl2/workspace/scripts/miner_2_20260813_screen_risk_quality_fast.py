"""miner_2 cycle-38 screening (fast vectorized): risk/quality/conditional-macro factor family.
Candidates (data visible at t, no lookahead):
  sharpe_20, updown_vol_ratio_60, overnight_mom_20, overnight_var_share_20,
  drawup_20, tail_ratio_20, dxy_up_ret_60, comm_beta_60
Metrics: daily cross-sectional Spearman IC vs h=10 fwd return; ICIR; hit; coverage; 10d rank turnover;
decay by horizon; max abs library correlation (npy artifacts only).
Gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import json, glob, os
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

DATE_PATH = "../persistent/date.json"
date_state = json.load(open(DATE_PATH))
TRADING_DAYS = date_state["trading_days"]
VISIBLE = date_state["visible_through"]
ROW0 = TRADING_DAYS.index("2020-01-01")
ROW1 = TRADING_DAYS.index(VISIBLE)
GRID = TRADING_DAYS[ROW0:ROW1 + 1]
GIDX = {d: i for i, d in enumerate(GRID)}
print(f"grid rows: {len(GRID)}  {GRID[0]}..{GRID[-1]}")

acct = get_account_dict()
ASSETS = list(acct.get("watch_list", []))
HORIZON = 10
MIN_ASSETS = 8

def load_asset(sym):
    df = get_stock_daily_data(sym, days=2100)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

DATA = {s: load_asset(s) for s in ASSETS}
print("loaded", len([k for k, v in DATA.items() if v is not None]), "assets")

dxy = pd.read_csv("../persistent/index_data/DXY.csv", parse_dates=["date"])
dxy["date"] = pd.to_datetime(dxy["date"]).dt.strftime("%Y-%m-%d")
dxy = dxy.set_index("date")["close"].astype(float)
dxy_ret = dxy.pct_change().reindex(GRID)

def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) < 1e-12, np.nan, a / b)

# ---- compute factor arrays per asset (aligned to asset's own rows) ----
F = {}   # sym -> dict of factor arrays (numpy, same length as asset df)
POS = {} # sym -> {date_str: row_pos}
for s, df in DATA.items():
    if df is None or len(df) < 100:
        continue
    close = df["close"].values.astype(float)
    ret = np.full(len(close), np.nan)
    ret[1:] = close[1:] / close[:-1] - 1.0
    o = df["open"].values.astype(float)
    gap = np.full(len(close), np.nan)
    gap[1:] = o[1:] / close[:-1] - 1.0
    intra = safe_div(close, o) - 1.0
    d = {}
    # 1. sharpe_20
    def roll_mean(x, w):
        c = np.cumsum(np.where(np.isnan(x), 0.0, x)); n = np.cumsum(~np.isnan(x))
        out = np.full(len(x), np.nan)
        out[w:] = (c[w:] - c[:-w]) / np.maximum(n[w:] - n[:-w], 1)
        return out
    def roll_std(x, w):
        mu = roll_mean(x, w)
        sq = roll_mean(x * x, w)
        return np.sqrt(np.maximum(sq - mu * mu, 0.0))
    mu20 = roll_mean(ret, 20); sd20 = roll_std(ret, 20)
    d["sharpe_20"] = safe_div(mu20, sd20)
    # 2. updown_vol_ratio_60
    up = np.where(ret > 0, ret, np.nan); dn = np.where(ret < 0, ret, np.nan)
    ups = roll_std(up, 60); dns = roll_std(dn, 60)
    d["updown_vol_ratio_60"] = safe_div(ups, dns)
    # 3. overnight_mom_20
    d["overnight_mom_20"] = roll_mean(gap, 20)
    # 4. overnight_var_share_20
    vo = roll_mean(gap * gap, 20) - roll_mean(gap, 20) ** 2
    vi = roll_mean(intra * intra, 20) - roll_mean(intra, 20) ** 2
    d["overnight_var_share_20"] = safe_div(vo, vo + vi)
    # 5. drawup_20: rolling max drawup from running 20d low-base
    d["drawup_20"] = np.full(len(close), np.nan)
    for i in range(19, len(close)):
        seg = close[max(0, i - 19):i + 1]
        d["drawup_20"][i] = float(np.max(seg / seg[0] - 1.0))
    # 6. tail_ratio_20
    d["tail_ratio_20"] = np.full(len(ret), np.nan)
    for i in range(19, len(ret)):
        seg = ret[max(0, i - 19):i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 10:
            continue
        p90, p50, p10 = np.percentile(seg, [90, 50, 10])
        dd = p50 - p10
        if abs(dd) > 1e-12:
            d["tail_ratio_20"][i] = (p90 - p50) / dd
    # 7. dxy_up_ret_60
    dxyv = dxy_ret.reindex(df.index).values
    upm = dxyv > 0; dnm = dxyv < 0
    d["dxy_up_ret_60"] = np.full(len(ret), np.nan)
    for i in range(59, len(ret)):
        seg = ret[i - 59:i + 1]; um = upm[i - 59:i + 1]; dm = dnm[i - 59:i + 1]
        ru = seg[um].mean(); rd = seg[dm].mean()
        if np.isnan(ru) or np.isnan(rd):
            continue
        d["dxy_up_ret_60"][i] = ru - rd
    # 8. comm_beta_60
    if "WTI" in DATA and DATA["WTI"] is not None and "COPPER" in DATA and DATA["COPPER"] is not None:
        bw = DATA["WTI"]["close"].pct_change().reindex(df.index).values
        bc = DATA["COPPER"]["close"].pct_change().reindex(df.index).values
        basket = np.nanmean(np.vstack([bw, bc]), axis=0)
        d["comm_beta_60"] = np.full(len(ret), np.nan)
        for i in range(59, len(ret)):
            x = basket[i - 59:i + 1]; y = ret[i - 59:i + 1]
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() < 20:
                continue
            xx = x[m]; yy = y[m]; vx = xx.var()
            if vx > 1e-12:
                d["comm_beta_60"][i] = float(np.cov(xx, yy)[0, 1] / vx)
    F[s] = d
    POS[s] = {dts: i for i, dts in enumerate(df.index)}
print("factor arrays for", len(F), "assets")

names = ["sharpe_20", "updown_vol_ratio_60", "overnight_mom_20", "overnight_var_share_20",
         "drawup_20", "tail_ratio_20", "dxy_up_ret_60", "comm_beta_60"]

# ---- precompute per-asset position and fwd return over GRID ----
POSG = {}  # sym -> np array of grid positions (nan if missing)
FWD = {}   # sym -> np array of h=10 fwd returns on grid
for s in ASSETS:
    if s not in F:
        continue
    df = DATA[s]
    close = df["close"].values.astype(float)
    pg = np.full(len(GRID), -1, dtype=int)
    fwd = np.full(len(GRID), np.nan)
    for t in GRID:
        if t in POS[s]:
            i = POS[s][t]
            pg[GIDX[t]] = i
            if i + HORIZON < len(close):
                fwd[GIDX[t]] = close[i + HORIZON] / close[i] - 1.0
    POSG[s] = pg
    FWD[s] = fwd

def ic_series(fname, h=HORIZON):
    ics = np.full(len(GRID), np.nan)
    for gi in range(len(GRID) - h):
        xs, ys = [], []
        for s in ASSETS:
            if s not in F or fname not in F[s]:
                continue
            i = POSG[s][gi]
            if i < 0 or np.isnan(F[s][fname][i]):
                continue
            j = i + h
            close = DATA[s]["close"].values
            if j >= len(close):
                continue
            x = F[s][fname][i]; y = close[j] / close[i] - 1.0
            if not np.isnan(y):
                xs.append(x); ys.append(y)
        if len(xs) >= MIN_ASSETS:
            ics[gi] = np.corrcoef(pd.Series(xs).rank().values, pd.Series(ys).rank().values)[0, 1]
    return ics

def turnover_10d(fname):
    diffs = []
    prev = None
    for gi in range(len(GRID)):
        vals = {}
        for s in ASSETS:
            if s in F and fname in F[s]:
                i = POSG[s][gi]
                if i >= 0 and not np.isnan(F[s][fname][i]):
                    vals[s] = F[s][fname][i]
        if len(vals) < MIN_ASSETS:
            prev = None
            continue
        rk = pd.Series(vals).rank(pct=True)
        if prev is not None:
            common = prev.index.intersection(rk.index)
            if len(common) >= MIN_ASSETS:
                diffs.append(float((rk[common] - prev[common]).abs().mean()))
        prev = rk
    return float(np.mean(diffs)) if diffs else np.nan

# library artifacts for corr audit (npy only, fast)
def load_library_artifacts():
    mats = {}
    for f in glob.glob("factors/*.signal.npy"):
        base = os.path.basename(f)[: -len(".signal.npy")]
        try:
            a = np.load(f, allow_pickle=True)
            mats[base] = np.asarray(a, dtype=float)
        except Exception as e:
            print("  skip artifact", f, e)
    return mats

def max_lib_corr(sigmat, mats):
    best, best_name = 0.0, None
    for name, m in mats.items():
        if m.shape != sigmat.shape:
            continue
        mask = ~(np.isnan(sigmat) | np.isnan(m))
        if mask.sum() < 500:
            continue
        a = sigmat[mask]; b = m[mask]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        if abs(rho) > best:
            best, best_name = abs(rho), name
    return best, best_name

MATS = load_library_artifacts()
print("library artifacts for corr audit:", len(MATS), list(MATS.keys()))

print("\n=== SCREENING RESULTS (h=10 Spearman daily IC; gates |IC|>=0.0070 |ICIR|>=0.0840) ===")
summary = {}
for nm in names:
    ic = ic_series(nm)
    ic = ic[~np.isnan(ic)]
    if len(ic) < 200:
        print(f"{nm:24s} insufficient IC dates ({len(ic)})")
        continue
    mean_ic = float(ic.mean()); std_ic = float(ic.std())
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    hit = float((ic > 0).mean())
    # coverage
    covs = []
    for gi in range(0, len(GRID), 5):
        nv = sum(1 for s in ASSETS if s in F and nm in F[s] and POSG[s][gi] >= 0 and not np.isnan(F[s][nm][POSG[s][gi]]))
        covs.append(nv / len(ASSETS))
    cov = float(np.mean(covs)) if covs else np.nan
    to10 = turnover_10d(nm)
    # decay
    decay = {}
    for h in [1, 3, 5, 10, 15, 20]:
        ih = ic_series(nm, h)
        ih = ih[~np.isnan(ih)]
        if len(ih) > 100:
            decay[str(h)] = round(float(ih.mean()), 4)
    # max lib corr
    sig = np.full((len(GRID), len(ASSETS)), np.nan)
    for ai, s in enumerate(ASSETS):
        if s in F and nm in F[s]:
            for gi in range(len(GRID)):
                i = POSG[s][gi]
                if i >= 0:
                    sig[gi, ai] = F[s][nm][i]
    mcorr, mname = max_lib_corr(sig, MATS)
    pass_gate = (abs(mean_ic) >= 0.0070) and (abs(icir) >= 0.0840)
    print(f"{nm:24s} IC={mean_ic:+.4f} ICIR={icir:+.3f} hit={hit:.3f} n={len(ic)} cov={cov:.3f} to10={to10:.3f} maxLibCorr={mcorr:.3f}({mname}) PASS={pass_gate}")
    print(f"  decay(h): {decay}")
    # regime split: first half vs last half of sample
    mid = len(ic) // 2
    ic1, ic2 = ic[:mid], ic[mid:]
    print(f"  reg: early IC={ic1.mean():+.4f} ICIR={ic1.mean()/ic1.std() if ic1.std()>0 else np.nan:+.3f} (n={len(ic1)}) | late IC={ic2.mean():+.4f} ICIR={ic2.mean()/ic2.std() if ic2.std()>0 else np.nan:+.3f} (n={len(ic2)})")
    summary[nm] = {"ic": round(mean_ic, 4), "icir": round(icir, 3), "hit": round(hit, 3),
                   "n_dates": len(ic), "coverage": round(cov, 3), "turnover_10d": round(to10, 3),
                   "max_lib_corr": round(mcorr, 3), "max_lib_corr_name": mname,
                   "pass": pass_gate, "decay": decay,
                   "regime_early": {"ic": round(float(ic1.mean()), 4), "icir": round(float(ic1.mean() / ic1.std()), 3) if ic1.std() > 0 else None},
                   "regime_late": {"ic": round(float(ic2.mean()), 4), "icir": round(float(ic2.mean() / ic2.std()), 3) if ic2.std() > 0 else None}}

json.dump(summary, open("scripts/_miner2_20260813_screen_results.json", "w"), indent=1)
print("\nsaved: scripts/_miner2_20260813_screen_results.json")

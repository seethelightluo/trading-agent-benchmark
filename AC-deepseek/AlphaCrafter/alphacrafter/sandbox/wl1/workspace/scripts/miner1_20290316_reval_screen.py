"""miner_1 cycle 2029-03-16: re-validate library factors + screen new candidates.
Fast numpy Spearman (rank then Pearson per date). Gates: abs IC>=0.0070, abs ICIR>=0.0840 at 10d.
"""
import json
import numpy as np
import pandas as pd

VISIBLE = "2029-03-15"
ASSETS = ["000300.SH", "000688.SH", "BTC", "CN10Y", "COPPER", "ETH", "HSI",
          "N225", "NDX", "SOX", "SPX", "SX5E", "US10Y", "WTI", "XAU"]
OBS = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
GATE_IC, GATE_ICIR = 0.0070, 0.0840


def trading_days():
    d = json.load(open("../persistent/date.json"))
    return [x for x in d["trading_days"] if x <= VISIBLE]


def load_prices():
    days = trading_days(); idx = pd.Index(days, name="date"); out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[df["date"] <= VISIBLE].set_index("date").reindex(idx).ffill()
        close = df["close"].astype(float)
        out[a] = pd.DataFrame({
            "close": close, "ret": close.pct_change(),
            "high": df["high"].astype(float), "low": df["low"].astype(float),
            "open": df["open"].astype(float)}, index=idx)
    return out


def load_obs():
    days = trading_days(); idx = pd.Index(days, name="date"); out = {}
    for o in OBS:
        df = pd.read_csv(f"../persistent/index_data/{o}.csv")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        out[o] = df["close"].astype(float).set_index("date").reindex(idx).ffill()
    return out


def factor_panel(fn, frames, obs=None, min_valid=8):
    cols = {}
    for a in ASSETS:
        try:
            s = fn(frames[a], obs, a)
        except Exception as e:
            s = pd.Series(np.nan, index=frames[a].index)
        cols[a] = pd.Series(s, index=frames[a].index).astype(float)
    p = pd.DataFrame(cols)
    gd = p.notna().sum(axis=1)
    return p, gd[gd >= min_valid].index


def _fast_ic(fv, fr, min_valid=8):
    """fv, fr: numpy arrays (dates x assets). Returns per-date spearman IC."""
    n = fv.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        a, b = fv[i], fr[i]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < min_valid:
            continue
        x, y = a[m], b[m]
        # rank
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        rx -= rx.mean(); ry -= ry.mean()
        den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        if den > 0:
            out[i] = float((rx * ry).sum() / den)
    return out


def ic_analysis(panel, good_dates, frames, horizons=(1, 5, 10), adm_horizon=10,
                min_valid=8, direction=1):
    P = panel.values
    res = {}
    for h in horizons:
        fwd = np.column_stack([frames[a]["close"].shift(-h).values / frames[a]["close"].values - 1.0
                               for a in ASSETS])
        pos = [panel.index.get_loc(dt) for dt in good_dates if dt in panel.index]
        ics = _fast_ic(P, fwd, min_valid)[pos]
        ics = ics[np.isfinite(ics)]
        if len(ics) == 0:
            res[h] = {"ic": np.nan, "icir": np.nan, "hit": np.nan, "n": 0}
        else:
            icm = float(ics.mean()); icsd = float(ics.std(ddof=1))
            res[h] = {"ic": icm, "icir": icm / icsd if icsd > 0 else 0.0,
                      "hit": float((ics * direction > 0).mean()), "n": int(len(ics))}
    cov = float(panel.notna().sum().sum() / (panel.shape[0] * panel.shape[1]))
    ranks = panel.rank(axis=1)
    to10 = float(ranks.diff().abs().mean(axis=1)[good_dates].mean() * 10.0)
    adm = res[adm_horizon]
    return {"by_horizon": {str(k): v for k, v in res.items()},
            "adm_ic": adm["ic"], "adm_icir": adm["icir"], "adm_hit": adm["hit"],
            "adm_n_dates": adm["n"], "coverage_asset_days": cov,
            "n_dates_ge8": int((panel.notna().sum(axis=1) >= min_valid).sum()),
            "n_total_dates": int(len(good_dates)), "turnover_10d_rank": to10}


def gate(m):
    return abs(m["adm_ic"]) >= GATE_IC and abs(m["adm_icir"]) >= GATE_ICIR


def show(tag, m):
    flag = "PASS" if gate(m) else "fail"
    print(f"[{tag}] {flag} adm10 IC={m['adm_ic']:+.4f} ICIR={m['adm_icir']:+.3f} "
          f"hit={m['adm_hit']:.3f} n={m['adm_n_dates']} cov={m['coverage_asset_days']:.3f} "
          f"to10={m['turnover_10d_rank']:.2f}", flush=True)
    for h in sorted(m["by_horizon"], key=int):
        v = m["by_horizon"][h]
        print(f"    h={h:>2}: IC={v['ic']:+.4f} ICIR={v['icir']:+.3f} hit={v['hit']:.3f} n={v['n']}", flush=True)


frames = load_prices(); obs = load_obs()
print(f"loaded {len(frames['SPX'])} days through {VISIBLE}", flush=True)

# ---- 1) re-validate library effective factors ----
def f_mom(df, o, a): return df["close"].shift(5) / df["close"].shift(125) - 1.0
def f_vov(df, o, a):
    return df["ret"].rolling(20).std() / df["ret"].rolling(60).std() - 1.0
def f_vixb(df, o, a):
    vix = o["VIX"]; dvi = vix.diff()
    beta = df["ret"].rolling(60).cov(dvi) / dvi.rolling(60).var()
    return beta * (vix.diff(20) > 0).astype(float)

print("\n===== RE-VALIDATION of effective library factors =====", flush=True)
lib_panels = {}
for fid, fn in [("mom_120d_skip5", f_mom), ("vol_of_vol20x60", f_vov),
                ("vix_beta_cond_60x20", f_vixb)]:
    p, gd = factor_panel(fn, frames, obs); lib_panels[fid] = p
    for tag, start in [("FULL", "2020-01-01"), ("ONLINE", "2026-07-16"), ("RECENT28", "2028-06-01")]:
        sub = p.index >= start; sg = gd[gd >= start]
        show(f"{fid} | {tag}", ic_analysis(p[sub], sg, frames))

# ---- 2) candidate screen ----
print("\n===== CANDIDATE SCREEN =====", flush=True)
C = {
 "risk_adj_mom_120x20": lambda df, o, a: (df["close"].shift(5) / df["close"].shift(125) - 1.0) / df["ret"].rolling(20).std(),
 "trend_mom_120x20": lambda df, o, a: (df["close"].shift(5) / df["close"].shift(125) - 1.0) * (df["close"] > df["close"].rolling(20).mean()).astype(float),
 "dd60_high": lambda df, o, a: df["close"] / df["close"].rolling(60).max() - 1.0,
 "range_pos_20": lambda df, o, a: (df["close"] - df["low"].rolling(20).min()) / (df["high"].rolling(20).max() - df["low"].rolling(20).min()) - 0.5,
 "skew_20": lambda df, o, a: df["ret"].rolling(20).skew(),
 "dist_ma200": lambda df, o, a: df["close"] / df["close"].rolling(200).mean() - 1.0,
 "mom5_volscaled": lambda df, o, a: (df["close"].shift(1) / df["close"].shift(6) - 1.0) / df["ret"].rolling(20).std(),
 "rev2_volscaled": lambda df, o, a: -(df["close"] / df["close"].shift(2) - 1.0) / df["ret"].rolling(20).std(),
}
def c_rate(df, o, a):
    du = frames["US10Y"]["close"].diff()
    beta = df["ret"].rolling(60).cov(du) / du.rolling(60).var()
    return beta * (frames["US10Y"]["close"].diff(20) > 0).astype(float)
def c_dxy(df, o, a):
    dd = o["DXY"].diff()
    beta = df["ret"].rolling(60).cov(dd) / dd.rolling(60).var()
    return beta * (o["DXY"].diff(20) > 0).astype(float)
def c_vov_hvix(df, o, a):
    vov = df["ret"].rolling(20).std() / df["ret"].rolling(60).std() - 1.0
    return vov * (o["VIX"] > 20).astype(float)
C["rate_beta_cond_60x20"] = c_rate
C["dxy_beta_cond_60x20"] = c_dxy
C["vov_highvix"] = c_vov_hvix

# C3 cross-sectional relative strength z-score 60d (panel-level, needs all assets)
def c3_panel(frames, obs):
    cols = {}
    for a in ASSETS:
        r60 = frames[a]["close"] / frames[a]["close"].shift(60) - 1.0
        cols[a] = r60
    rs = pd.DataFrame(cols)
    z = (rs - rs.mean(axis=1)) / rs.std(axis=1)
    return z, z.notna().sum(axis=1)

res = {}
for name, fn in C.items():
    p, gd = factor_panel(fn, frames, obs)
    m = ic_analysis(p, gd, frames)
    res[name] = m
    show(name, m)

zp, zg = c3_panel(frames, obs)
zg = zg[zg >= 8].index
res["cs_rs_z60"] = ic_analysis(zp, zg, frames)
show("cs_rs_z60", res["cs_rs_z60"])

print("\n===== ONLINE WINDOW (2026-07-16..) for all candidates =====", flush=True)
for name, fn in C.items():
    p, gd = factor_panel(fn, frames, obs)
    sub = p.index >= "2026-07-16"; sg = gd[gd >= "2026-07-16"]
    m = ic_analysis(p[sub], sg, frames)
    if gate(m):
        print(f"ONLINE PASS {name}: IC={m['adm_ic']:+.4f} ICIR={m['adm_icir']:+.3f} n={m['adm_n_dates']}", flush=True)
sub = zp.index >= "2026-07-16"; sg = zg[zg >= "2026-07-16"]
m = ic_analysis(zp[sub], sg, frames)
if gate(m):
    print(f"ONLINE PASS cs_rs_z60: IC={m['adm_ic']:+.4f} ICIR={m['adm_icir']:+.3f} n={m['adm_n_dates']}", flush=True)

json.dump({k: {"adm_ic": v["adm_ic"], "adm_icir": v["adm_icir"], "adm_hit": v["adm_hit"],
               "adm_n_dates": v["adm_n_dates"], "cov": v["coverage_asset_days"],
               "to10": v["turnover_10d_rank"], "by_horizon": v["by_horizon"]}
           for k, v in res.items()},
          open("scripts/miner1_20290316_screen_results.json", "w"), indent=1)
print("\nsaved.", flush=True)

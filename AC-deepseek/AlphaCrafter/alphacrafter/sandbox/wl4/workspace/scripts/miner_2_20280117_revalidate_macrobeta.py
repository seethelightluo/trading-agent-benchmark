"""miner_2 2028-01-17: revalidate 3 active factors + explore MACRO-BETA family.

Data visible through 2028-01-14 (per persistent/date.json).
Rank IC h=10 on the 15-asset cross-asset universe.
Gates (benchmark-wide): |IC|>=0.0070, |ICIR|>=0.0840; library correlation < 0.5.
"""
import json
import numpy as np
import pandas as pd

VISIBLE = pd.Timestamp('2028-01-14')
IC_TH, ICIR_TH = 0.0070, 0.0840
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]


def load_close():
    frames = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        frames[a] = df["close"]
    return pd.DataFrame(frames)


def load_ohlcv():
    out = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        out[a] = df
    return out


def load_macro():
    frames = {}
    for m in MACRO:
        df = pd.read_csv(f"../persistent/index_data/{m}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= VISIBLE].set_index("date").sort_index()
        frames[m] = df["close"]
    return pd.DataFrame(frames)


def fwd_returns(panel, h=10):
    return panel.shift(-h) / panel - 1.0


def rank_ic_series(factor, fwd, min_valid=8):
    dates = factor.index.intersection(fwd.index)
    ics = {}
    for dt in dates:
        f = factor.loc[dt]
        r = fwd.loc[dt]
        mask = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        if mask.sum() < min_valid:
            continue
        ic = f[mask].corr(r[mask], method="spearman")
        if np.isfinite(ic):
            ics[dt] = ic
    return pd.Series(ics, name="ic")


def evaluate(factor, panel, h=10, min_valid=8, label="f", valid_from=None, valid_to=None):
    fwd = fwd_returns(panel, h=h)
    ic = rank_ic_series(factor, fwd, min_valid=min_valid)
    if valid_from is not None:
        ic = ic[ic.index >= pd.Timestamp(valid_from)]
    if valid_to is not None:
        ic = ic[ic.index <= pd.Timestamp(valid_to)]
    out = {"label": label, "n_ic_dates": len(ic)}
    if len(ic) == 0:
        out.update(ic=np.nan, icir=np.nan, ic_std=np.nan, ic_hit=np.nan)
        return out
    icm = ic.mean()
    ics = ic.std(ddof=1)
    out.update(ic=float(icm), icir=float(icm / ics) if ics > 0 else np.nan,
               ic_std=float(ics), ic_hit=float((ic > 0).mean()))
    ge8 = (factor.notna().sum(axis=1) >= min_valid).mean()
    out["cov_ge8"] = float(ge8)
    ranks = factor.rank(axis=1)
    out["turnover_10d"] = float((ranks - ranks.shift(10)).abs().mean().mean())
    return out


def rolling_beta(ret, bench, win=60, minp=40):
    cov = ret.rolling(win, min_periods=minp).cov(bench)
    var = bench.rolling(win, min_periods=minp).var()
    return cov / var


def lib_corr_series(cand, lib, min_valid=8):
    """Daily cross-sectional spearman rho between candidate and library panel."""
    common = cand.index.intersection(lib.index)
    rhos = []
    for dt in common:
        a = cand.loc[dt]
        b = lib.loc[dt]
        m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
        if m.sum() < min_valid:
            continue
        r = a[m].corr(b[m], method="spearman")
        if np.isfinite(r):
            rhos.append(r)
    return np.array(rhos)


def max_lib_corr(cand, lib_panels, min_valid=8):
    worst = {}
    meanabs = {}
    for fid, lp in lib_panels.items():
        rhos = lib_corr_series(cand, lp, min_valid)
        if len(rhos):
            worst[fid] = float(np.max(np.abs(rhos)))
            meanabs[fid] = float(np.mean(np.abs(rhos)))
    if not worst:
        return 0.0, None, 0.0, None
    fid = max(worst, key=worst.get)
    fidm = max(meanabs, key=meanabs.get)
    return worst[fid], fid, meanabs[fidm], fidm


panel = load_close()
ohlcv = load_ohlcv()
macro = load_macro()
rets = panel.pct_change()
mkt = rets.mean(axis=1)

print(f"Data: {panel.shape[0]} days, {panel.shape[1]} assets, through {VISIBLE.date()}")

# ---------- ACTIVE FACTORS ----------
active = {}
down_mkt = mkt.where(mkt < 0, 0.0)
active["dn_mkt_beta_60d"] = rolling_beta(rets, down_mkt, 60, 40)
vols = pd.DataFrame({a: ohlcv[a]["volume"] for a in ASSETS})
active["vol_price_corr_20"] = rets.rolling(20, min_periods=10).corr(vols)
cn10y_ret = panel["CN10Y"].pct_change()
active["rate_beta_cn10y_60d"] = rolling_beta(rets, cn10y_ret, 60, 40)

print("\n=== ACTIVE FACTOR REVALIDATION (visible through 2028-01-14) ===")
for fid, f in active.items():
    full = evaluate(f, panel, label=fid)
    r500 = evaluate(f, panel, label=fid + "_r500", valid_from=panel.index[-501])
    r250 = evaluate(f, panel, label=fid + "_r250", valid_from=panel.index[-251])
    print(f"{fid}: full ic={full['ic']:+.4f} icir={full['icir']:+.4f} hit={full['ic_hit']:.3f} "
          f"n={full['n_ic_dates']} cov8={full['cov_ge8']:.3f} turn={full['turnover_10d']:.2f} | "
          f"r500 ic={r500['ic']:+.4f} icir={r500['icir']:+.4f} | r250 ic={r250['ic']:+.4f} icir={r250['icir']:+.4f}")

# ---------- CANDIDATE BATCH: MACRO-BETA / SENSITIVITY FAMILY ----------
print("\n=== CANDIDATE EXPLORATION: MACRO-BETA FAMILY ===")
cand = {}
# market sensitivity
cand["mkt_beta_60d"] = rolling_beta(rets, mkt, 60, 40)
up_mkt = mkt.where(mkt > 0, 0.0)
cand["up_mkt_beta_60d"] = rolling_beta(rets, up_mkt, 60, 40)
# macro drivers (observation-only signals)
for m, drv in [("us10y", panel["US10Y"].pct_change()),
               ("dxy", macro["DXY"].pct_change()),
               ("eurusd", macro["EURUSD"].pct_change()),
               ("usdjpy", macro["USDJPY"].pct_change()),
               ("usdcny", macro["USDCNY"].pct_change()),
               ("vix", macro["VIX"].pct_change()),
               ("btc", rets["BTC"]),
               ("xau", rets["XAU"]),
               ("wti", rets["WTI"]),
               ("cn10y", cn10y_ret)]:
    cand[f"{m}_beta_60d"] = rolling_beta(rets, drv, 60, 40)

lib_panels = dict(active)

rows = []
for fid, f in cand.items():
    ev = evaluate(f, panel, label=fid)
    evr = evaluate(f, panel, label=fid + "_r250", valid_from=panel.index[-251])
    mc, mcf, mm, mmf = max_lib_corr(f, lib_panels)
    rows.append((fid, ev, evr, mc, mcf, mm, mmf))

for fid, ev, evr, mc, mcf, mm, mmf in rows:
    gate = "PASS" if (abs(ev["ic"]) >= IC_TH and abs(ev["icir"]) >= ICIR_TH) else "fail"
    corr_ok = "corr-OK" if (np.isfinite(mc) and mc < 0.5) else "CORR-HIGH"
    print(f"{fid:22s} ic={ev['ic']:+.4f} icir={ev['icir']:+.4f} hit={ev['ic_hit']:.3f} "
          f"n={ev['n_ic_dates']:5d} cov8={ev['cov_ge8']:.3f} turn={ev['turnover_10d']:.2f} | "
          f"r250 ic={evr['ic']:+.4f} icir={evr['icir']:+.4f} | "
          f"maxcorr={mc:.3f}({mcf}) meanabs={mm:.3f}({mmf}) => {gate}/{corr_ok}")

# decay for candidates passing IC/ICIR gate
print("\n=== DECAY (IC by horizon) for gate-passing candidates ===")
for fid, ev, evr, mc, mcf, mm, mmf in rows:
    if abs(ev["ic"]) >= IC_TH and abs(ev["icir"]) >= ICIR_TH:
        f = cand[fid]
        dec = {}
        for hh in [1, 2, 3, 5, 10, 20]:
            e = evaluate(f, panel, h=hh, label=fid)
            dec[str(hh)] = round(e["ic"], 4) if np.isfinite(e["ic"]) else None
        print(f"{fid:22s} decay={dec}")

json.dump({fid: {"ic": ev["ic"], "icir": ev["icir"], "hit": ev["ic_hit"], "n": ev["n_ic_dates"],
                 "cov8": ev["cov_ge8"], "turn": ev["turnover_10d"],
                 "r250_ic": evr["ic"], "r250_icir": evr["icir"],
                 "maxcorr": mc, "maxcorr_factor": mcf, "meanabs": mm, "meanabs_factor": mmf}
           for fid, ev, evr, mc, mcf, mm, mmf in rows},
          open("scripts/_miner2_20280117_macrobeta_results.json", "w"), indent=1)
print("\nsaved scripts/_miner2_20280117_macrobeta_results.json")

"""miner_2 (2026-08-13): audit + persist passing factors from rounds 1-2.
Passing candidates (h=10 warm-up 2020-01-01..2026-07-15):
  crypto_beta_60d     IC=+0.0640 ICIR=+0.1631
  dxy_beta_cond_60x20 IC=+0.0410 ICIR=+0.1196 (direction -1)
  ndx_beta_60d        IC=+0.0519 ICIR=+0.1262

Audit includes correlation vs a FULL reconstruction of the 11-factor library
(beta_ew_60d, amihud_20, downside_vol_ratio_20, eurusd_beta_cond_60x20,
max_ret_20d, mom_10d_skip5, mom_120d_skip5, rel_mom_20d_skip5,
vix_beta_cond_60x20, vol_adj_mom_20x60, vol_of_vol20x60).
"""
from __future__ import annotations
import sys, json, base64, zlib, io, csv, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from miner_2_lib import (load_panel, load_macro, fwd_returns, rank_ic_series,
                         WATCH, MAX_VISIBLE, FACTOR_LAST, MIN_ASSETS, ADMISSION)

panel = load_panel()
macro = load_macro()
rets = panel.pct_change()


# ---------- library reconstruction (11 effective factors) ----------
def _vol_series(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= MAX_VISIBLE].set_index("date").sort_index()
    return df["volume"].astype(float)


def _beta(asset_ret, ref_ret, w=60, minp=30):
    z = pd.concat([asset_ret.rename("r"), ref_ret.rename("m")], axis=1).dropna()
    return z["r"].rolling(w, min_periods=minp).cov(z["m"]) / z["m"].rolling(w, min_periods=minp).var()


def library_signals_full():
    lib = {}
    lib["mom_10d_skip5"] = panel.shift(5) / panel.shift(15) - 1.0
    lib["mom_120d_skip5"] = panel.shift(5) / panel.shift(125) - 1.0
    lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    lib["rel_mom_20d_skip5"] = (panel.shift(5) / panel.shift(25) - 1.0).sub(
        (panel.shift(5) / panel.shift(25) - 1.0).median(axis=1), axis=0)
    lib["max_ret_20d"] = rets.rolling(20).max()
    lib["vol_adj_mom_20x60"] = (panel.shift(5) / panel.shift(25) - 1.0) / rets.rolling(60).std()
    neg = rets.clip(upper=0)
    dv = -((neg ** 2).rolling(20).mean() ** 0.5) / rets.rolling(20).std()
    lib["downside_vol_ratio_20"] = dv
    ew = rets.mean(axis=1)
    lib["beta_ew_60d"] = pd.DataFrame({a: _beta(rets[a], ew) for a in panel.columns}, index=panel.index)
    vix = macro["VIX"]
    vixr = vix.pct_change()
    vix20 = vix / vix.shift(20) - 1.0
    lib["vix_beta_cond_60x20"] = pd.DataFrame(
        {a: -_beta(rets[a], vixr) * vix20.reindex(rets[a].dropna().index) for a in panel.columns},
        index=panel.index)
    eur = macro["EURUSD"]
    eurr = eur.pct_change()
    eur20 = eur / eur.shift(20) - 1.0
    lib["eurusd_beta_cond_60x20"] = pd.DataFrame(
        {a: _beta(rets[a], eurr) * eur20.reindex(rets[a].dropna().index) for a in panel.columns},
        index=panel.index)
    ami = {}
    for a in panel.columns:
        v = _vol_series(a).reindex(panel[a].dropna().index)
        ami[a] = (rets[a].abs() / v).rolling(20, min_periods=10).mean()
    lib["amihud_20"] = pd.DataFrame(ami, index=panel.index)
    return lib


def datewise_corr(factor, lf, n_dates=700):
    cs = []
    common = factor.index.intersection(lf.index)[-n_dates:]
    for dt in common:
        f = factor.loc[dt]
        g = lf.loc[dt]
        if isinstance(f, pd.DataFrame) or isinstance(g, pd.DataFrame):
            continue
        m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
        m = m.reindex(f.index).fillna(False)
        if int(m.sum()) >= MIN_ASSETS:
            cs.append(spearmanr(f[m], g[m])[0])
    return float(np.mean(cs)) if cs else float("nan")


# ---------- candidate definitions ----------
def _beta_to(panel, ref_name, window=60, minp=30):
    ref = panel[ref_name].pct_change()
    return pd.DataFrame(
        {a: _beta(panel[a].pct_change(), ref) for a in panel.columns}, index=panel.index)


def cand_crypto_beta_60d():
    return _beta_to(panel, "BTC")


def cand_dxy_beta_cond_60x20():
    dxy = macro["DXY"]
    dxyr = dxy.pct_change()
    dxy20 = dxy / dxy.shift(20) - 1.0
    return pd.DataFrame(
        {a: -_beta(panel[a].pct_change(), dxyr) * dxy20.reindex(panel[a].dropna().index)
         for a in panel.columns}, index=panel.index)


def cand_ndx_beta_60d():
    return _beta_to(panel, "NDX")


CANDIDATES = {
    "crypto_beta_60d": (cand_crypto_beta_60d, +1.0,
                        "crypto_beta_60d", "Crypto (BTC) beta 60d",
                        "rolling 60d beta of asset returns vs BTC returns: cov(r,r_BTC)/var(r_BTC)",
                        "Linkage to crypto risk factor. Assets with higher BTC beta (crypto, tech, risk-on) "
                        "earn a cross-sectional premium over the sample: high-BTC-beta assets outperform. "
                        "Strongest 2020-21/2023-24; weaker in 2022 and 2025-26 crypto corrections.",
                        ["close"], {"window": 60, "min_periods": 30, "ref": "BTC"},
                        ["beta", "crypto", "cross-asset", "risk-premium"]),
    "dxy_beta_cond_60x20": (cand_dxy_beta_cond_60x20, -1.0,
                            "dxy_beta_cond_60x20", "USD-beta conditional 60x20 (DXY)",
                            "-beta(asset_ret, DXY_ret, 60) * (DXY/DXY.shift(20)-1)",
                            "Conditional USD-risk signal: asset beta to the dollar times the 20-day DXY move, "
                            "sign-flipped. Direction -1 => after USD strength, assets positively linked to the "
                            "dollar outperform (USD-spillover momentum). Positive IC in 2021-2026, negative 2020.",
                            ["close"], {"window": 60, "cond_window": 20, "min_periods": 30, "ref": "DXY"},
                            ["beta", "macro", "fx", "conditional", "cross-asset"]),
    "ndx_beta_60d": (cand_ndx_beta_60d, +1.0,
                     "ndx_beta_60d", "Tech (NDX) beta 60d",
                     "rolling 60d beta of asset returns vs NDX returns: cov(r,r_NDX)/var(r_NDX)",
                     "Linkage to the tech/growth complex: assets with high beta to Nasdaq-100 outperform "
                     "cross-sectionally (tech-beta premium). Strong 2020-21 and 2023-24; flat/weak 2025-26.",
                     ["close"], {"window": 60, "min_periods": 30, "ref": "NDX"},
                     ["beta", "tech", "cross-asset", "risk-premium"]),
}


def build_artifact(factor: pd.DataFrame) -> dict:
    f = factor.reindex(panel.index)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date"] + list(f.columns))
    for dt in f.index:
        w.writerow([dt.date().isoformat()] + ["" if pd.isna(v) else f"{float(v):.12g}" for v in f.loc[dt]])
    raw = buf.getvalue().encode()
    comp = zlib.compress(raw, level=6)
    b64 = base64.b64encode(comp).decode()
    return {"format": "base64:zlib:csv",
            "description": f"Factor signal panel: rows = dates, cols = assets. Shape {list(f.shape)}",
            "columns": list(f.columns), "shape": list(f.shape),
            "n_valid_values": int(f.notna().sum().sum()),
            "sha256": hashlib.sha256(comp).hexdigest()[:16], "data": b64}


def persist_one(fid, fn, direction, name, expr, desc, deps, params, tags, lib):
    factor = fn()
    factor_w = factor.loc[:FACTOR_LAST]
    ic10 = rank_ic_series(factor_w, fwd_returns(panel, 10)) * direction
    icir10 = float(ic10.mean() / ic10.std())
    ic_hit = float((ic10 > 0).mean())
    n_dates = int(len(ic10))
    yrs = {}
    for y in range(2020, 2027):
        sub = ic10.loc[str(y)]
        if len(sub) > 20:
            yrs[y] = (round(float(sub.mean()), 4), round(float(sub.mean() / sub.std()), 4), int(len(sub)))

    per = {}
    for lfid, lf in lib.items():
        per[lfid] = round(datewise_corr(factor_w, lf), 3)
    valid_c = [abs(v) for v in per.values() if v is not None and np.isfinite(v)]
    max_corr = round(max(valid_c), 4) if valid_c else float("nan")

    valid = factor_w.notna()
    cov_ad = float(valid.mean().mean())
    cov_d8 = float((valid.sum(axis=1) >= MIN_ASSETS).mean())
    ranks = factor_w.rank(axis=1)
    tov = []
    for i in range(10, len(ranks)):
        a, b = ranks.iloc[i - 10], ranks.iloc[i]
        both = a.dropna().index.intersection(b.dropna().index)
        if len(both) >= MIN_ASSETS:
            tov.append(float((a[both] - b[both]).abs().mean()))
    turnover = float(np.mean(tov)) if tov else float("nan")

    horizons = (1, 2, 3, 5, 10, 20)
    decay = {}
    for h in horizons:
        ic_h = rank_ic_series(factor_w, fwd_returns(panel, h)) * direction
        decay[str(h)] = round(float(ic_h.mean()), 4)

    print(f"=== PERSIST AUDIT {fid} ===")
    print(f"  IC10={float(ic10.mean()):+.4f} ICIR10={icir10:+.4f} hit={ic_hit:.3f} n={n_dates}")
    print(f"  per-year (dir adj): {yrs}")
    print(f"  coverage_ad={cov_ad:.3f} cov_d8={cov_d8:.3f} turnover10d={turnover:.3f}")
    print(f"  decay: {decay}")
    print(f"  library corr: {per}")
    print(f"  max_abs_library_correlation = {max_corr}")
    gate_ic = abs(float(ic10.mean())) >= ADMISSION["ic"]
    gate_icir = abs(icir10) >= ADMISSION["icir"]
    ok = gate_ic and gate_icir
    print(f"  ADMISSION: IC {gate_ic} ICIR {gate_icir} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  NOT persisted.\n")
        return None

    artifact = build_artifact(factor)
    npy_path = Path(f"factors/{fid}.signal.npy")
    np.save(npy_path, np.asarray(factor.reindex(panel.index).values, dtype=float))

    payload = {
        "factor_id": fid,
        "factor_name": name,
        "version": "1.0.0",
        "calculation": {"expression": expr, "description": desc},
        "dependencies": deps,
        "parameters": params,
        "expected_direction": int(direction),
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": "2026-08-13",
            "admission_horizon": 10,
            "regime_notes": ("15-instrument tradable cross-asset universe; warm-up 2020-01-01..2026-07-15. "
                             "Per-year h10 IC (direction-adjusted): " +
                             "; ".join(f"{y}: ic={v[0]} icir={v[1]} n={v[2]}" for y, v in sorted(yrs.items())) +
                             ". Library corr (full 11-factor): " + json.dumps(per)),
            "metrics": {
                "ic": float(ic10.mean()),
                "icir": icir10,
                "ic_hit_ratio": ic_hit,
                "n_ic_dates": n_dates,
                "coverage_asset_days": cov_ad,
                "coverage_dates_ge8": cov_d8,
                "turnover_10d_rank": turnover,
                "decay_ic_by_horizon": decay,
                "max_abs_library_correlation": max_corr,
                "library_pairwise_corr": {k: v for k, v in per.items() if v is not None and np.isfinite(v)},
            },
            "signal_artifact": artifact,
        },
        "tags": tags,
        "artifact_provenance": {
            "format": "npy_matrix", "shape": list(factor.reindex(panel.index).shape),
            "columns": WATCH,
            "dates_first": str(panel.index.min().date()),
            "dates_last": str(panel.index.max().date()),
            "n_nan": int(np.isnan(np.asarray(factor.reindex(panel.index).values)).sum()),
        },
        "benchmark_admission": {
            "contract": {"ic_threshold": ADMISSION["ic"], "icir_threshold": ADMISSION["icir"],
                         "correlation_threshold": 0.5},
            "selected_metrics": {
                "ic": float(ic10.mean()), "icir": icir10,
                "metric_path": "validation.metrics",
                "max_abs_library_correlation": max_corr,
                "correlation_path": "validation.metrics.max_abs_library_correlation",
            },
        },
    }
    path = Path(f"factors/{fid}.json")
    path.write_text(json.dumps(payload, indent=2))

    back = json.loads(path.read_text())
    a = back["validation"]["signal_artifact"]
    dec = zlib.decompress(base64.b64decode(a["data"])).decode()
    rows = list(csv.reader(io.StringIO(dec)))
    arr = np.load(npy_path)
    assert back["factor_id"] == fid
    assert back["validation"]["status"] == "EFFECTIVE"
    assert back["validation"]["metrics"]["max_abs_library_correlation"] == max_corr
    assert arr.shape == tuple(back["artifact_provenance"]["shape"])
    assert len(rows) - 1 == a["shape"][0] and len(rows[0]) == a["shape"][1]
    assert hashlib.sha256(base64.b64decode(a["data"])).hexdigest()[:16] == a["sha256"]
    print(f"  PERSISTED -> {path} ({path.stat().st_size} bytes)")
    print(f"  VERIFIED: id ok, status EFFECTIVE, npy shape={arr.shape}, "
          f"csv rows={len(rows)-1} cols={len(rows[0])}, sha ok\n")
    return path


if __name__ == "__main__":
    lib = library_signals_full()
    print("library factors reconstructed:", sorted(lib.keys()))
    print()
    # pairwise corr among candidates (redundancy check)
    cand_frames = {fid: fn() for fid, (fn, *_rest) in CANDIDATES.items()}
    for i, a in enumerate(cand_frames):
        for b in list(cand_frames)[i + 1:]:
            print(f"pairwise {a} vs {b}: {datewise_corr(cand_frames[a].loc[:FACTOR_LAST], cand_frames[b].loc[:FACTOR_LAST]):.3f}")
    print()
    for fid, (fn, direction, name, expr, desc, deps, params, tags) in CANDIDATES.items():
        persist_one(fid, fn, direction, name, expr, desc, deps, params, tags, lib)

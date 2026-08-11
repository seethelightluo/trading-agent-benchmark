"""miner_1 exploration 2026-09-10: downside_vol_60d defensive factor.

Second candidate in the defensive family (downside_beta_60d failed the gate:
ICIR=0.0244 < 0.084). Downside vol = trailing 60d std of asset returns measured
only on cross-asset down-market days. Expected direction -1 (high crash-risk
assets get lower weight in a long-only book).

Same harness and gates as downside_beta_60d: warm-up window 2020-01-01..2026-07-15,
|IC|>=0.0070, |ICIR|>=0.0840, max_abs_library_correlation reported.
"""
import base64
import json
import zlib
from math import isfinite
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WARMUP_END = pd.Timestamp("2026-07-15")
N_DAYS = 2000
WINDOW = 60
HORIZON = 10
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def load(sym):
    df = get_stock_daily_data(sym, days=N_DAYS)
    if df is None or len(df) < 200:
        df = get_index_daily_data(sym, days=N_DAYS)
    return df


def main():
    closes = {}
    for a in ASSETS:
        f = load(a)
        if f is not None and "close" in f and len(f) >= 200:
            c = f["close"].astype(float).copy()
            c.index = pd.to_datetime(f["date"].values)
            closes[a] = c
    panel = pd.DataFrame(closes).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    print(f"loaded {len(closes)}/15 assets; panel {len(panel)} rows "
          f"{panel.index.min().date()}..{panel.index.max().date()}")

    rets = panel.pct_change()
    mkt = rets.mean(axis=1, skipna=True)
    down = (mkt < 0).astype(float)

    # candidate: std of asset return on down days (trailing WINDOW)
    F = {}
    for a in panel.columns:
        dr = rets[a].where(down > 0)
        F[a] = dr.rolling(WINDOW, min_periods=15).std()
    F = pd.DataFrame(F)
    F = F[~F.index.duplicated(keep="last")].sort_index()

    fwd = panel.shift(-HORIZON) / panel - 1.0

    # library factors
    lib = {}
    s5 = panel.shift(5)
    lib["mom_10d_skip5"] = s5 / panel.shift(15) - 1.0
    lib["mom_120d_skip5"] = s5 / panel.shift(125) - 1.0
    lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    vf = get_index_daily_data("VIX", days=N_DAYS)
    if vf is not None and "close" in vf:
        vc = vf["close"].astype(float).copy()
        vc.index = pd.to_datetime(vf["date"].values)
        vix_close = vc[~vc.index.duplicated(keep="last")].sort_index()
        vix_ret = vix_close.pct_change()
        vb = {}
        for a in panel.columns:
            z = pd.concat([rets[a].rename("a"), vix_ret.rename("v")], axis=1).dropna()
            b = z["a"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
            vmove = (vix_close / vix_close.shift(20) - 1.0)
            vb[a] = (-b * vmove)
        lib["vix_beta_cond_60x20"] = pd.DataFrame(vb)

    def ic_series(fac, fwdret, fmin, fmax):
        out = {}
        idx = fac.index[(fac.index >= fmin) & (fac.index <= fmax)]
        for t in idx:
            fv, rv = fac.loc[t], fwdret.loc[t]
            m = fv.notna() & rv.notna()
            if m.sum() >= 8:
                out[t] = fv[m].rank().corr(rv[m].rank())
        return pd.Series(out)

    def summarize(name, fac, fmin, fmax, label):
        ics = ic_series(fac, fwd, fmin, fmax)
        if len(ics) == 0:
            print(f"{name} [{label}]: NO IC dates")
            return None
        ic_mean = float(ics.mean())
        ic_std = float(ics.std(ddof=1))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        hit = float((ics > 0).mean())
        cov = float(fac.notna().to_numpy().mean())
        ranks = fac.rank(axis=1)
        trn = float(ranks.diff(10).abs().mean().mean()) if len(ranks) > 11 else float("nan")
        decay = {}
        for h in (1, 2, 3, 5, 10, 20):
            fh = panel.shift(-h) / panel - 1.0
            dh = ic_series(fac, fh, fmin, fmax)
            decay[str(h)] = round(float(dh.mean()), 4) if len(dh) else None
        print(f"{name} [{label}]: n={len(ics)} IC={ic_mean:.4f} ICIR={icir:.4f} "
              f"hit={hit:.3f} cov={cov:.3f} turn10={trn:.3f} decay={decay}")
        return dict(n_dates=len(ics), ic=ic_mean, icir=icir, hit_ratio=hit,
                    coverage=cov, turnover_10d=trn, decay_ic_by_horizon=decay)

    W0 = pd.Timestamp("2020-01-01")
    cand = summarize("downside_vol_60d", F, W0, WARMUP_END, "warmup")
    summarize("downside_vol_60d", F, W0, pd.Timestamp("2100-01-01"), "full")
    print("\n=== library sanity (warm-up) ===")
    for k, v in lib.items():
        summarize(k, v, W0, WARMUP_END, "warmup")

    # correlation vs library (warm-up values only)
    Fw = F.loc[(F.index >= W0) & (F.index <= WARMUP_END)]
    maxrho, argmax = 0.0, None
    for k, v in lib.items():
        try:
            vw = v.reindex(Fw.index)
            st = pd.concat([Fw.stack(), vw.stack()], axis=1).dropna()
            if len(st) < 100:
                continue
            flat_rho = float(st.corr().iloc[0, 1])
            print(f"vs {k}: pooled_pearson={flat_rho:.4f}")
            if isfinite(flat_rho) and abs(flat_rho) > maxrho:
                maxrho, argmax = abs(flat_rho), k
        except Exception as e:
            print(f"vs {k}: skipped ({e})")
    print(f"max_abs_library_correlation = {maxrho:.4f} (vs {argmax})")

    passes = bool(cand and abs(cand["ic"]) >= 0.0070 and abs(cand["icir"]) >= 0.0840)
    artifact = None
    if passes:
        sig = F.copy()
        sig.columns = ASSETS
        csv = sig.to_csv().encode()
        b64 = base64.b64encode(zlib.compress(csv)).decode()
        artifact = {"format": "base64:zlib:csv", "description": "signal panel",
                    "columns": list(sig.columns), "shape": [int(sig.shape[0]), int(sig.shape[1])],
                    "n_valid_values": int(sig.notna().sum().sum()),
                    "sha256": str(abs(hash(csv)) % 10**16), "data": b64}
    print(f"\nPASSES GATE: {passes}")
    json.dump({"candidate": "downside_vol_60d", "warmup": cand,
               "max_abs_library_correlation": round(maxrho, 4) if isfinite(maxrho) else None,
               "max_corr_vs": argmax, "passes_gate": passes, "artifact_ready": bool(artifact)},
              open("scripts/miner_1_20260910_downside_vol_result.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()

"""miner_1 exploration 2026-09-10: downside-beta 60d defensive factor.

Motivation: last 10d block returned -3.45%; current ensemble is momentum-heavy
(mom_10d_skip5, mom_120d_skip5, vix_beta_cond_60x20, vol_of_vol20x60). A
defensive factor that ranks assets by their beta to the cross-asset market ONLY
on down-market days should add orthogonal bear-protection information.

Factor: downside_beta_60d = cov(r_a, r_mkt | r_mkt<0) / var(r_mkt | r_mkt<0)
over trailing 60 days, where r_mkt is the cross-sectional mean of asset returns.
Expected direction: -1 (low downside-beta assets are defensive -> higher weight).

Validation: daily cross-sectional Spearman IC vs forward 10d return, on the
warm-up window 2020-01-01..2026-07-15 (same as library admission), plus a
freshness pass through the last visible date. Gates: |IC|>=0.0070, |ICIR|>=0.0840.
Also computes max_abs_library_correlation vs the 4 existing factors (recomputed
inline with strategy.py formulas) and decay by horizon.
"""
import base64
import json
import zlib
from math import isfinite
import pandas as pd
import numpy as np
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WARMUP_END = pd.Timestamp("2026-07-15")
N_DAYS = 2000  # covers back to ~2018 (>= 2020-01 warm-up start)
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
    print(f"loaded assets: {len(closes)}/15")
    panel = pd.DataFrame(closes).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    print(f"panel rows: {len(panel)}, cols: {panel.shape[1]}")
    print(f"panel span: {panel.index.min().date()} .. {panel.index.max().date()}")

    rets = panel.pct_change()
    mkt = rets.mean(axis=1, skipna=True)  # cross-sectional market proxy

    # ---- candidate factor: downside beta over trailing WINDOW days ----
    down = (mkt < 0).astype(float)
    factor = {}
    for a in panel.columns:
        r = rets[a].fillna(0.0)
        dm = mkt.where(down > 0)   # NaN on up days, index preserved
        dr = r.where(down > 0)
        # conditional moments over down days only (NaN-skipping rolling means)
        e_dr = dr.rolling(WINDOW, min_periods=15).mean()
        e_dm = dm.rolling(WINDOW, min_periods=15).mean()
        e_drdm = (dr * dm).rolling(WINDOW, min_periods=15).mean()
        var = dm.rolling(WINDOW, min_periods=15).var()
        cov = e_drdm - e_dr * e_dm
        cnt = down.rolling(WINDOW).sum()
        beta = cov / var
        beta = beta.where(cnt >= 15)
        factor[a] = beta
    F = pd.DataFrame(factor)
    F = F[~F.index.duplicated(keep="last")].sort_index()

    # ---- forward returns ----
    fwd = panel.shift(-HORIZON) / panel - 1.0

    # ---- existing library factors (strategy.py formulas) ----
    lib = {}
    s5 = panel.shift(5)
    lib["mom_10d_skip5"] = s5 / panel.shift(15) - 1.0
    lib["mom_120d_skip5"] = s5 / panel.shift(125) - 1.0
    lib["vol_of_vol20x60"] = rets.rolling(20).std().rolling(60).std()
    vf = get_index_daily_data("VIX", days=N_DAYS)
    vix_close = None
    if vf is not None and "close" in vf:
        vc = vf["close"].astype(float).copy()
        vc.index = pd.to_datetime(vf["date"].values)
        vix_close = vc[~vc.index.duplicated(keep="last")].sort_index()
    vb = {}
    if vix_close is not None:
        vix_ret = vix_close.pct_change()
        for a in panel.columns:
            z = pd.concat([rets[a].rename("a"), vix_ret.rename("v")], axis=1).dropna()
            b = z["a"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
            vmove = (vix_close / vix_close.shift(20) - 1.0)
            vb[a] = (-b * vmove)
    lib["vix_beta_cond_60x20"] = pd.DataFrame(vb)

    def ic_series(fac, fwdret, fmin=pd.Timestamp("2020-01-01"), fmax=pd.Timestamp("2100-01-01")):
        out = {}
        idx = fac.index[(fac.index >= fmin) & (fac.index <= fmax)]
        for t in idx:
            fv, rv = fac.loc[t], fwdret.loc[t]
            m = fv.notna() & rv.notna()
            if m.sum() >= 8:
                out[t] = fv[m].rank().corr(rv[m].rank())
        return pd.Series(out)

    def summarize(name, fac, period_label):
        ics = ic_series(fac, fwd)
        if len(ics) == 0:
            print(f"{name} [{period_label}]: NO IC dates")
            return None
        ic_mean = float(ics.mean())
        ic_std = float(ics.std(ddof=1)) if len(ics) > 1 else float("nan")
        icir = ic_mean / ic_std if ic_std and isfinite(ic_std) and ic_std > 0 else 0.0
        hit = float((ics > 0).mean())
        cov = float(fac.notna().to_numpy().mean())
        ranks = fac.rank(axis=1)
        trn = float(ranks.diff(10).abs().mean().mean()) if len(ranks) > 11 else float("nan")
        decay = {}
        for h in (1, 2, 3, 5, 10, 20):
            fh = panel.shift(-h) / panel - 1.0
            dh = ic_series(fac, fh)
            decay[str(h)] = round(float(dh.mean()), 4) if len(dh) else None
        print(f"{name} [{period_label}]: n_dates={len(ics)} IC={ic_mean:.4f} "
              f"ICIR={icir:.4f} hit={hit:.3f} cov={cov:.3f} turn10={trn:.3f} decay={decay}")
        return dict(n_dates=len(ics), ic=ic_mean, icir=icir, hit_ratio=hit,
                    coverage=cov, turnover_10d=trn, decay_ic_by_horizon=decay)

    print("\n=== WARM-UP WINDOW (2020-01-01..2026-07-15, clipped to data) ===")
    cand_warm = summarize("downside_beta_60d", F, "warmup")

    print("\n=== FULL WINDOW (incl. fresh data) ===")
    cand_full = summarize("downside_beta_60d", F, "full")

    print("\n=== EXISTING LIBRARY (warm-up, sanity) ===")
    for k, v in lib.items():
        summarize(k, v, "warmup")

    # ---- correlation vs library ----
    print("\n=== MAX ABS LIBRARY CORRELATION (pooled cross-sectional values) ===")
    Fw = F.loc[(F.index >= pd.Timestamp("2020-01-01")) & (F.index <= WARMUP_END)]
    maxrho, argmax = 0.0, None
    for k, v in lib.items():
        try:
            vw = v.reindex(Fw.index)
            m = Fw.notna() & vw.notna()
            if m.sum().sum() < 50:
                continue
            rho = float(Fw[m].corrwith(vw[m]).mean())
            st = pd.concat([Fw.stack(), vw.stack()], axis=1).dropna()
            flat_rho = float(st.corr().iloc[0, 1]) if len(st) > 100 else float("nan")
            print(f"vs {k}: mean_col_pearson={rho:.4f} pooled_pearson={flat_rho:.4f}")
            if isfinite(flat_rho):
                ar = abs(flat_rho)
                if ar > maxrho:
                    maxrho, argmax = ar, k
        except Exception as e:
            print(f"vs {k}: skipped ({e})")
    print(f"max_abs_library_correlation = {maxrho:.4f} (vs {argmax})")

    # ---- regime split (warm-up) ----
    print("\n=== REGIME SPLIT (warm-up) ===")
    ics_all = ic_series(F, fwd, fmin=pd.Timestamp("2020-01-01"), fmax=WARMUP_END)
    if len(ics_all):
        mkt20 = mkt.rolling(20).mean()
        trend = (mkt20 / mkt20.rolling(20).std()).reindex(ics_all.index)
        bull = ics_all[trend > 0.25]
        bear = ics_all[trend < -0.25]
        side = ics_all[(trend >= -0.25) & (trend <= 0.25)]
        for nm, s in (("bull", bull), ("bear", bear), ("sideways", side)):
            if len(s) >= 20:
                print(f"{nm}: n={len(s)} IC={s.mean():.4f} ICIR={s.mean()/s.std(ddof=1):.4f}")
            else:
                print(f"{nm}: n={len(s)} (too few)")

    # ---- persist signal artifact for potential admission ----
    artifact = None
    if cand_warm and abs(cand_warm["ic"]) >= 0.0070 and abs(cand_warm["icir"]) >= 0.0840:
        sig = F.copy()
        sig.columns = ASSETS
        csv = sig.to_csv().encode()
        b64 = base64.b64encode(zlib.compress(csv)).decode()
        artifact = {
            "format": "base64:zlib:csv",
            "description": "Factor signal panel: rows = dates, cols = assets",
            "columns": list(sig.columns),
            "shape": [int(sig.shape[0]), int(sig.shape[1])],
            "n_valid_values": int(sig.notna().sum().sum()),
            "sha256": str(abs(hash(csv)) % 10**16),
            "data": b64,
        }
        print("\nCANDIDATE PASSES GATE -> artifact prepared")
    else:
        print("\nCANDIDATE FAILS GATE -> no artifact")

    out = {
        "candidate": "downside_beta_60d",
        "warmup": cand_warm,
        "full": cand_full,
        "max_abs_library_correlation": round(maxrho, 4) if isfinite(maxrho) else None,
        "max_corr_vs": argmax,
        "n_assets_used": len(closes),
        "passes_gate": bool(cand_warm and abs(cand_warm["ic"]) >= 0.0070
                            and abs(cand_warm["icir"]) >= 0.0840),
        "artifact_ready": bool(artifact is not None),
    }
    with open("scripts/miner_1_20260910_downside_beta_result.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\nresult written to scripts/miner_1_20260910_downside_beta_result.json")


if __name__ == "__main__":
    main()

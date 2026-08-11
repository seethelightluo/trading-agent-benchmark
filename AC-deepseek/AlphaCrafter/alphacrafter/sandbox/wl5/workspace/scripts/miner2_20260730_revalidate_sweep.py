"""miner_2: 3-month re-validation sweep of the live factor library (2026-07-30 cycle).

For every live factor in factors/ (non-bak, non-ensemble):
  - decode/recompute signal artifact
  - full-window IC/ICIR (h=10) + recent 3-month IC/ICIR (last ~63 trading days)
  - regime ICs and drift flag
Signals are recovered from persisted artifacts where available; otherwise recomputed
from the factor's calculation expression via known builders.
"""
import sys, os, json, io, base64, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split, WATCH)

VIS = "2026-07-29"
H = 10
LIB_DIR = "factors"

close = closes_panel(VIS)
idx = close.index
fr = forward_returns(close, H)
ret = close.pct_change()

# ---------- artifact decoding ----------
def decode_artifact(meta):
    a = meta.get("validation", {}).get("signal_artifact")
    if not a:
        return None
    dec = zlib.decompress(base64.b64decode(a["data"])).decode("utf-8")
    sig = pd.read_csv(io.StringIO(dec), index_col=0, parse_dates=True)
    return sig.reindex(columns=close.columns).reindex(close.index)


# ---------- expression-based rebuilders (for factors w/o artifacts) ----------
def rebuild(meta):
    fid = meta["factor_id"]
    expr = meta.get("calculation", {}).get("expression", "")
    if fid == "trend_r2_30_signed":
        # signed R^2 of 30d linear trend on log price
        out = {}
        for a in close.columns:
            c = close[a].dropna()
            if len(c) < 40:
                out[a] = pd.Series(np.nan, index=idx)
                continue
            lp = np.log(c)
            x = np.arange(30)
            def r2(seg):
                y = seg.values
                slope = np.polyfit(x, y, 1)[0]
                pred = np.polyval(np.polyfit(x, y, 1), x)
                ss_res = float(np.sum((y - pred) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2v = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                return r2v * np.sign(slope)
            out[a] = lp.rolling(30).apply(r2, raw=False).reindex(idx)
        return pd.DataFrame(out, index=idx)
    return None


# ---------- library scan ----------
print(f"--- LIBRARY RE-VALIDATION SWEEP (VIS={VIS}, h={H}) ---")
rows = []
for fn in sorted(os.listdir(LIB_DIR)):
    if not fn.endswith(".json") or fn == "factor_ensemble.json":
        continue
    with open(os.path.join(LIB_DIR, fn), encoding="utf-8") as f:
        meta = json.load(f)
    fid = meta["factor_id"]
    sig = decode_artifact(meta)
    src = "artifact"
    if sig is None:
        sig = rebuild(meta)
        src = "rebuilt"
    if sig is None:
        print(f"{fid:24s} no artifact and no rebuild path -> skipped")
        continue
    sig = sig.reindex(columns=close.columns).reindex(close.index)
    ic = ic_series(sig, fr, min_valid=8)
    m = summary_metrics(ic, sig, fr, close, h=H)
    if m is None:
        print(f"{fid:24s} insufficient IC dates")
        continue
    # recent 3-month window (last 63 trading days with IC)
    icr = ic.dropna()
    ic_recent = icr[icr.index >= icr.index[-1] - pd.Timedelta(days=95)]  # ~63 trading days
    r_ic = float(ic_recent.mean()) if len(ic_recent) >= 20 else float("nan")
    r_std = float(ic_recent.std(ddof=1)) if len(ic_recent) >= 21 else float("nan")
    r_icir = float(r_ic / r_std) if r_std and np.isfinite(r_std) and r_std > 0 else float("nan")
    reg = regime_split(ic)
    drift = "DRIFT?" if (np.isfinite(r_icir) and abs(r_icir) < 0.084 and abs(m["icir"] or 0) >= 0.084) else "ok"
    rows.append((fid, m, r_ic, r_icir, len(ic_recent), reg, drift, src, meta))
    print(f"{fid:24s} full ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']:4d} | recent3m ic={r_ic:+.4f} icir={r_icir:+.4f} n={len(ic_recent):3d} "
          f"| regimes { {k: round(v['ic'],3) for k,v in reg.items()} } | {drift} [{src}]")

print("\n--- FACTORS WITH RECENT-3M |IC|>=0.007 AND |ICIR|>=0.084 (still effective) ---")
for fid, m, r_ic, r_icir, n, reg, drift, src, meta in rows:
    if np.isfinite(r_icir) and abs(r_ic) >= 0.007 and abs(r_icir) >= 0.084:
        print(f"  {fid:24s} recent ic={r_ic:+.4f} icir={r_icir:+.4f} n={n}")

print("\n--- FACTORS WITH RECENT-3M DECAY / REGIME REVERSAL ---")
for fid, m, r_ic, r_icir, n, reg, drift, src, meta in rows:
    sign_full = np.sign(m["ic"])
    sign_recent = np.sign(r_ic) if np.isfinite(r_ic) else 0
    if sign_full != sign_recent:
        print(f"  {fid:24s} SIGN FLIP full={m['ic']:+.4f} -> recent={r_ic:+.4f}")

# save summary
out = {fid: {"full": {k: m[k] for k in ("ic", "icir", "ic_hit_ratio", "n_ic_dates")},
             "recent_3m_ic": round(r_ic, 4) if np.isfinite(r_ic) else None,
             "recent_3m_icir": round(r_icir, 4) if np.isfinite(r_icir) else None,
             "regime": {k: v["ic"] for k, v in reg.items()},
             "source": src}
       for fid, m, r_ic, r_icir, n, reg, drift, src, meta in rows}
with open("scripts/miner2_20260730_revalidate_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, default=str)
print("\nsaved -> scripts/miner2_20260730_revalidate_results.json")

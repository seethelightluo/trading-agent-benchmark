"""Enumerate recoverable library signals in factors/ and their status."""
import os, json, base64, gzip, zlib, pickle
import numpy as np
import pandas as pd

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
close = cache["close"]
idx = close.index


def load_lib_factor(path):
    d = json.load(open(path))
    sa = d.get("signal_artifact")
    if sa is None:
        v = d.get("validation") or {}
        sa = v.get("signal_artifact") or (v.get("metrics") or {}).get("signal_artifact")
    if isinstance(sa, str):
        p = os.path.join("factors", sa)
        if p.endswith(".npy") and os.path.exists(p):
            return pd.DataFrame(np.load(p), index=idx, columns=SYMBOLS)
        return None
    if isinstance(sa, dict):
        fmt = str(sa.get("format", ""))
        if "gzip" in fmt and "n_dates" in sa:
            try:
                raw = base64.b64decode(sa["data"])
                try:
                    raw = gzip.decompress(raw)
                except Exception:
                    raw = zlib.decompress(raw)
                arr = np.frombuffer(raw, dtype="<f4").reshape(sa["n_dates"], sa["n_symbols"])
                start = pd.Timestamp(sa["date_start"]); end = pd.Timestamp(sa["date_end"])
                mask = (idx >= start) & (idx <= end) & close.notna().all(axis=1)
                common = idx[mask]
                n = min(len(common), arr.shape[0])
                dates = common[-n:]
                return pd.DataFrame(arr[-n:], index=dates, columns=sa.get("symbols", SYMBOLS))
            except Exception as e:
                print("  gzip decode fail", e)
        if "base64:zlib:csv" in fmt or ("data" in sa and "n_dates" not in sa):
            try:
                raw = zlib.decompress(base64.b64decode(sa["data"]))
                txt = raw.decode("utf-8", errors="replace")
                rows = []
                for line in txt.splitlines():
                    if not line.strip() or line.startswith("date,"):
                        continue
                    parts = line.split(",")
                    rows.append((parts[0], [float(v) if v not in ("", "NA") else np.nan for v in parts[1:]]))
                df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows],
                                  columns=sa.get("columns", SYMBOLS))
                df.index = pd.to_datetime(df.index)
                return df
            except Exception as e:
                print("  csv decode fail", e)
    return None


lib = {}
for f in sorted(os.listdir("factors")):
    if not f.endswith(".json") or f.endswith(".bak"):
        continue
    df = load_lib_factor(os.path.join("factors", f))
    if df is not None and len(df) > 100:
        lib[f] = df
print("Recoverable library signals:", len(lib))
for k, v in lib.items():
    st = "?"
    try:
        st = json.load(open(os.path.join("factors", k))).get("validation", {}).get("status", "?")
    except Exception:
        pass
    print(f"  {k:48s} {v.shape} {st}")

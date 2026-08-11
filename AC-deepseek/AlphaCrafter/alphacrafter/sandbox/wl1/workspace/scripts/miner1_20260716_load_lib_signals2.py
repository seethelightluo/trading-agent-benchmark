"""miner1: decode ALL effective library factor signal artifacts into one aligned matrix.
Handles both artifact formats:
  - 'data_b64' : base64 gzip float32 matrix (miner2 style, n_dates x n_symbols)
  - 'data'     : base64:zlib:csv (mom style)
Alignment: each artifact must cover the 15-symbol universe; we rebuild the date axis
from the first decodable artifact's CSV (or use a canonical date axis from raw data).
"""
import json, glob, re, base64, gzip, zlib
import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, "scripts")
from miner1_common import load_close, SYMBOLS, CUT

SNAP = re.compile(r"\.\d{8}T\d{9,12}\.json$")

def base_name(bn):
    m = SNAP.search(bn)
    return bn[:m.start()] + ".json" if m else bn

def decode_artifact(d):
    art = d.get("signal_artifact") or d.get("artifact") or {}
    if not art:
        return None, None
    if "data_b64" in art:
        raw = base64.b64decode(art["data_b64"])
        M = np.frombuffer(gzip.decompress(raw), dtype=np.float32)
        M = M.reshape(art["n_dates"], art["n_symbols"])
        symbols = art.get("symbols")
        dates = None
        if "date_start" in art and "n_dates" in art:
            dates = pd.date_range(art["date_start"], periods=art["n_dates"], freq="D")
        return M, (dates, symbols)
    if "data" in art and art.get("format", "").startswith("base64"):
        comp = art["data"].split(":", 2)[2]
        csv = zlib.decompress(base64.b64decode(comp)).decode()
        df = pd.read_csv(pd.io.common.StringIO(csv))
        return df.values.astype(np.float32), (pd.to_datetime(df.iloc[:, 0]), list(df.columns[1:]))
    return None, None

# collect latest file per base name
files = {}
for f in glob.glob("factors/*.json"):
    if ".bak" in f or "/evicted/" in f:
        continue
    bn = f.split("/")[-1]
    files.setdefault(base_name(bn), []).append(f)

lib = {}
for base, lst in sorted(files.items()):
    # lst contains full paths like factors/xxx.json ; base is the bare filename
    if any(os.path.basename(p) == base for p in lst):
        path = next(p for p in lst if os.path.basename(p) == base)
    else:
        path = max(lst)
    try:
        d = json.load(open(path))
    except Exception as e:
        print("ERR load", base, e); continue
    v = d.get("validation", {})
    if v.get("status") != "EFFECTIVE":
        continue
    M, meta = decode_artifact(d)
    if M is None:
        print("NO DECODABLE ARTIFACT:", base)
        continue
    dates, symbols = meta
    if dates is None:
        print("NO DATES:", base); continue
    lib[base] = (M, dates, symbols)
    print(f"  {base}: shape={M.shape} dates={dates[0].date()}..{dates[-1].date()} symbols={len(symbols)}")

print("decoded library factors:", len(lib))
closes = load_close()
canon = closes[SYMBOLS[0]].index
print("canonical axis from raw data:", len(canon), canon[0].date(), canon[-1].date())

ordered = []
names = []
for base, (M, dates, symbols) in sorted(lib.items()):
    if M.shape[1] != len(SYMBOLS):
        print("SKIP symbol mismatch:", base, M.shape)
        continue
    if len(dates) != len(canon):
        # reindex onto canonical axis if possible (dates are daily)
        if not M.shape[0] == len(dates):
            print("SKIP date mismatch:", base, M.shape, len(dates)); continue
        print("NOTE date count differs from canon; keeping own axis:", base)
    ordered.append(M)
    names.append(base)

if ordered:
    mat = np.stack(ordered, axis=0)  # n_factors x n_dates x n_symbols
    np.save("scripts/_lib_signal_matrix.npy", mat.astype(np.float32))
    with open("scripts/_lib_signal_names.txt", "w") as f:
        f.write("\n".join(names))
    print("saved lib matrix:", mat.shape, "names:", names)

"""miner1: decode all effective library factor signal artifacts into one aligned matrix.
Library files may have timestamped snapshot copies; prefer base names (no timestamp).
"""
import json, glob, re, base64, gzip
import numpy as np
import pandas as pd

SNAP = re.compile(r"\.\d{8}T\d{9,12}\.json$")

def load_artifact(path):
    d = json.load(open(path))
    art = d.get("signal_artifact")
    if not art or "data_b64" not in art:
        return None
    M = np.frombuffer(gzip.decompress(base64.b64decode(art["data_b64"])), dtype=np.float32)
    M = M.reshape(art["n_dates"], art["n_symbols"])
    return d, M

def base_name(bn):
    m = SNAP.search(bn)
    return bn[:m.start()] + ".json" if m else bn

# collect latest file per base name
files = {}
for f in glob.glob("factors/*.json"):
    if ".bak" in f or "reason" in f or "/evicted/" in f:
        continue
    bn = f.split("/")[-1]
    files.setdefault(base_name(bn), []).append(f)

lib = {}
for base, lst in files.items():
    # prefer exact base, else latest snapshot
    if base in lst:
        path = "factors/" + base
    else:
        path = max(lst)  # lexicographic max ~ latest timestamp
    d = json.load(open(path))
    v = d.get("validation", {})
    if v.get("status") != "EFFECTIVE":
        continue
    art = d.get("signal_artifact")
    if not art or "data_b64" not in art:
        print("NO ARTIFACT:", base)
        continue
    try:
        _, M = load_artifact(path)
    except Exception as e:
        print("ERR decode", base, e)
        continue
    lib[base] = M

print("library factors with artifacts:", len(lib))
for k in sorted(lib):
    print("  ", k, lib[k].shape)
np.save("scripts/_lib_signal_matrix.npy",
        np.array([lib[k] for k in sorted(lib)], dtype=np.float32))
with open("scripts/_lib_signal_names.txt", "w") as f:
    for k in sorted(lib):
        f.write(k + "\n")
# alignment info: first artifact provides dates
print("done")

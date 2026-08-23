"""Compute a library-correlation-aware candidate factor value on demand (no artifacts persisted)."""
import json
import numpy as np

def load_panel(fact_file):
    d = json.load(open(fact_file))
    sa = d['validation']['signal_artifact']
    import base64, zlib, io
    raw = base64.b64decode(sa['data'])
    txt = zlib.decompress(raw).decode('utf-8')
    import csv
    rows = list(csv.reader(io.StringIO(txt)))
    header = rows[0]
    cols = header
    dates = [r[0] for r in rows[1:]]
    M = np.array([[float(x) if x != '' else np.nan for x in r[1:]] for r in rows[1:]])
    return dates, cols[1:], M

print("skew_20d panel:", load_panel('factors/skew_20d.json')[0][-1], load_panel('factors/skew_20d.json')[2].shape)
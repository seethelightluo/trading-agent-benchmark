"""Probe: decode existing signal artifacts and compare with recomputed factor signals."""
import sys, io, zlib, base64, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from miner3_lib import load_panel, per_asset, WATCH

panel = load_panel(days=4000)
print('panel shape:', panel.shape, '| range:', panel.index.min(), '->', panel.index.max())
print('panel columns:', list(panel.columns))

def decode_artifact(d):
    sa = d['validation']['signal_artifact']
    raw = base64.b64decode(sa['data'])
    raw = zlib.decompress(raw)
    df = pd.read_csv(io.BytesIO(raw), index_col=0)
    return df

def encode_artifact(df):
    buf = io.BytesIO()
    df.to_csv(buf)
    comp = zlib.compress(buf.getvalue())
    return base64.b64encode(comp).decode()

for fid in ['mom_10d_skip5', 'mom_120d_skip5', 'vix_beta_cond_60x20', 'vol_of_vol20x60']:
    d = json.load(open(f'factors/{fid}.json'))
    df = decode_artifact(d)
    print(f'--- {fid}: decoded shape {df.shape}, cols {list(df.columns)[:5]}... n_valid {np.isfinite(df.values).sum()}')
    print('   head row0:', df.iloc[0].tolist()[:3])
    print('   tail row-1:', df.iloc[-1].tolist()[:3])

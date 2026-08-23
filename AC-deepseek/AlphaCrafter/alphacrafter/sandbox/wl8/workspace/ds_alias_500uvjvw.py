import pandas as pd, os
# check date ranges of evicted signal artifacts by decoding a small part
import json, base64, zlib, io
def load_sig(path):
    d=json.load(open(path))
    sa=d.get('validation',{}).get('signal_artifact',{})
    if not isinstance(sa,dict) or 'data' not in sa: return None
    raw=base64.b64decode(sa['data'])
    csv=zlib.decompress(raw).decode()
    from io import StringIO
    df=pd.read_csv(StringIO(csv))
    return df
for name in ['mom_10d_skip5','vix_beta_cond_60x20','yield_beta_cond_60x20']:
    try:
        df=load_sig(f'factors/evicted/{name}.json')
        if df is not None:
            print(name, df.shape, df.columns[0], 'rows with non-null in first col:', df.iloc[:,1].notna().sum())
    except Exception as e:
        print(name,'ERR',e)
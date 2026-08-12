import json, os, glob
# List all factor JSON files and their validation status
files = sorted(glob.glob('factors/*.json'))
print("=== FACTOR FILES (top level) ===")
for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
        status = d.get('validation', {}).get('status', '?')
        last = d.get('last_validated', d.get('validation', {}).get('last_validated', '?'))
        metrics = d.get('validation', {}).get('metrics', {})
        ic = metrics.get('ic', metrics.get('ic_r250', '?'))
        icir = metrics.get('icir', metrics.get('icir_r250', '?'))
        print(f"{os.path.basename(f):45s} status={status:12s} last={str(last)[:10]:10s} ic={ic} icir={icir}")
    except Exception as e:
        print(f, "ERR", e)

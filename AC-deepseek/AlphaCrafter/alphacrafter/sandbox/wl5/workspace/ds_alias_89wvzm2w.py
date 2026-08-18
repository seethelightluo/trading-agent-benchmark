
import json, glob, os
print("=== factor files (non-bak, non-dir) ===")
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        status = d.get('validation', {}).get('status', '?')
        last = d.get('last_validated', d.get('validation', {}).get('validated_at', '?'))
        ic = d.get('validation', {}).get('metrics', {}).get('ic')
        icir = d.get('validation', {}).get('metrics', {}).get('icir')
        print(f"{os.path.basename(f):45s} status={status:12s} last={str(last)[:19]:20s} ic={ic} icir={icir}")
    except Exception as e:
        print(f, "ERR", e)

import json, glob, os

files = sorted(glob.glob('factors/*.json'))
files = [f for f in files if not f.endswith('.bak') and '.npy' not in f]
print("total json files (incl duplicate timestamps):", len(files))

with open(files[0]) as fh:
    d = json.load(fh)
print("SAMPLE FILE:", os.path.basename(files[0]))
print("TOP KEYS:", list(d.keys()))

import json, glob
# Look at a revalidation results file to understand metrics format and what's been validated recently
files = glob.glob('scripts/miner_3_20340803_revalidate_results.json') + glob.glob('scripts/*revalidate_results*.json')
print(files[-3:])
d = json.load(open(files[-1]))
print(json.dumps(d, indent=1)[:3000])
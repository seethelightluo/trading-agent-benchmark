import os, glob
# find any output txt files
for f in glob.glob('*.txt') + glob.glob('logs/*'):
    print(f, os.path.getsize(f) if os.path.isfile(f) else '')

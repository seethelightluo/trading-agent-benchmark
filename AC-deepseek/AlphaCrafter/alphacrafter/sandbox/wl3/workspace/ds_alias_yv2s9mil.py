# check ds_alias files - probably data source aliases
for f in sorted(os.listdir('.')):
    if f.startswith('ds_alias'):
        print(f, open(f).read()[:200])
        print('---')
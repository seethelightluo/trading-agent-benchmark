lines = open('memory.txt').readlines()
print("total lines:", len(lines))
for l in lines[-8:]:
    print(l.rstrip()[:1500])
    print('---')
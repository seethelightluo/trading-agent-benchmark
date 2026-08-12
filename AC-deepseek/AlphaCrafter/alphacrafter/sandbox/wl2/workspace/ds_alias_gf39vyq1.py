txt = open('memory.txt').read().strip().split('\n')
print("total lines:", len(txt))
for line in txt[-6:]:
    print(line[:600])
    print('---')

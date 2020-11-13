import matplotlib.pyplot as plt


path = 'log_info_1654.txt'
f = open(path, 'r')
lines = f.readlines()
loss = []
epoch = []
i = 1
for line in lines:
    epoch.append(i)
    data = line.split(' ')
    s1 = float(data[-1].strip('\n'))
    loss.append(s1)
    i = i + 1
plt.plot(epoch, loss, 'k')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
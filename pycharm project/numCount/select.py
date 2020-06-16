import matplotlib.pyplot as plt
import numpy as np
import json


# 读取原数据
name = ['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz', 'tag']
fopen = open('../data/rawdata/count.json', 'r')
line = json.loads(fopen.readline())
fopen.close()

# 对每个轴原始数据进行画图
for i in range(6):
    data = np.array(line[name[i]])[50:1050]
    x = np.arange(0, 20, 0.02)
    plt.subplot('32'+str(i+1))
    plt.title(name[i])
    plt.plot(x, data)
plt.show()


from detecta import detect_peaks
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import json


# 设置带通滤波器参数
b, a = signal.butter(2, [0.001, 0.03],  'bandpass')
# 读取原数据
fopen = open('../data/rawdata/count.json', 'r')
line = json.loads(fopen.readline())
fopen.close()

# 获取accy轴数据并滤波
data = np.array(line['accy'])
data = signal.filtfilt(b, a, data)

# 检测波峰和波谷
num_peak = detect_peaks(data)
num_valley = detect_peaks(-data)

# 取波峰和波谷点位置的值
peak = np.zeros(len(num_peak))
valley = np.zeros(len(num_valley))
for i in range(len(num_peak)):
    peak[i] = data[num_peak[i]]
for i in range(len(num_valley)):
    valley[i] = data[num_valley[i]]

# 画出图像
x = np.arange(0, len(data)*2/100, 0.02)
plt.plot(x, data)
plt.plot(num_peak*2/100, peak, '*')
plt.plot(num_valley*2/100, valley, '*')
plt.title('Count Action Number By Peaks')
plt.show()

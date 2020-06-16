from scipy import signal
import numpy as np
import json
import csv


print('Preprocess Data')
# 打开原数据和将要写入的文件
name = ['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz', 'tag']
fopen = open('../data/rawdata/data.json', 'r')
for i1 in range(7):
    locals()[name[i1]] = open('../data/processdata/' + name[i1] + '.csv', 'w', newline='')
# 设置低通滤波器参数
b, a = signal.butter(8, 0.2, 'lowpass')

# 主函数
for line in fopen:
    line = json.loads(line)
    if line['type'] == 1:
        # 将六个轴数据分别写入csv
        for i2 in range(6):
            data = np.array(line[name[i2]])
            data = data[50:1050]
            filtedData = signal.filtfilt(b, a, data)
            # 设置50%的重叠率
            for i3 in range(19):
                temp = filtedData[i3 * 50: (i3 + 2) * 50]
                writer = csv.writer(locals()[name[i2]])
                writer.writerow(temp)
                if i2 == 0:
                    csv.writer(locals()[name[6]]).writerow(str(int(line['activity'])))
fopen.close()
print('Finish')

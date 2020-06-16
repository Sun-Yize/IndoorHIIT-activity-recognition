import pandas as pd
import numpy as np
import csv
import feature_test


# 读取feature文件的类
feature_class = feature_test.getFeature()
name = ['accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz']

# 六个轴分别计算特征
for i1 in range(6):
    fopen = open('../../data/featuredata/' + name[i1] + '.csv', 'w', newline='')
    data = pd.read_csv('../../data/processdata/' + name[i1] + '.csv',  header=None)
    # 将特征写入csv文件
    for i2 in range(len(data)):
        row = np.array(data.iloc[i2, :])
        outputs = feature_class.get_all_feature(row)
        csv.writer(fopen).writerow(outputs)
    fopen.close()

# 将六轴的特征合并后写入csv
fopen = open('../../data/featuredata/total.csv', 'w', newline='')
for i1 in range(6):
    locals()[name[i1]] = pd.read_csv('../../data/featuredata/' + name[i1] + '.csv',  header=None)
# 合并六个轴的数据
for i2 in range(len(locals()[name[0]])):
    total = np.append(accx.iloc[i2, :], accy.iloc[i2, :])
    total = np.append(total, accz.iloc[i2, :])
    total = np.append(total, gyrx.iloc[i2, :])
    total = np.append(total, gyry.iloc[i2, :])
    total = np.append(total, gyrz.iloc[i2, :])
    csv.writer(fopen).writerow(total)
fopen.close()

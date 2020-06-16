import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer


# 导入处理好文件
inputs = pd.read_csv('../../data/featuredata/total.csv', header=None)
inputs = Imputer().fit_transform(inputs)
inputs = StandardScaler().fit_transform(inputs)

# 列出每个特征
feature = ['min', 'max', 'mean', 'median', 'mad', 'std', 'skew', 'kurtosis', 'iqr', 'energy', 'wskew', 'wkurtosis']
featureList = np.zeros(len(feature))

# 逐个计算方程
for i in range(np.shape(inputs)[1]):
    var = np.var(inputs[:, i])
    # 设定阈值
    if var > 1:
        featureList[i % len(feature)] += 1

# 将结果打印出来
print('方差筛选后，六个轴符合阈值出现的次数:')
for i in range(len(featureList)):
    print(feature[i], '=', int(featureList[i]), end=',')

"""
比较 Support Vector Machine
     Random Forest
     Neural Network
     K Nearest Neighbors 四个算法的准确率
"""
import pandas as pd
import numpy as np
import time
import SVM
import RF
import NN
import KNN


time_start = time.time()
# 选取随机数种子
np.random.seed(seed=10)
# 读取数据
inputs = pd.read_csv('../../data/featuredata/total.csv', header=None)
outputs = pd.read_csv('../../data/processdata/tag.csv', header=None)
# 将数据顺序打乱
train = np.array(pd.concat([inputs, outputs], axis=1))
np.random.shuffle(train)
# 分成输入和结果
inputs = pd.DataFrame(train).iloc[:, :-1]
outputs = pd.DataFrame(train).iloc[:, -1]

# 对四个模型分别进行训练，并进行十折交叉验证
result1 = SVM.model(inputs, outputs)
result2 = NN.model(inputs, outputs)
result3 = RF.model(inputs, outputs)
result4 = KNN.model(inputs, outputs)

time_end = time.time()

# 输出各个算法交叉验证准确率的平均值
print('SVM: mean accuracy = ', result1)
print('NN: mean accuracy = ', result2)
print('RF: mean accuracy = ', result3)
print('KNN: mean accuracy = ', result4)
print('Running time: {:.2f} Seconds'.format(time_end-time_start))
